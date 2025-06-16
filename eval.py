import os
import sys
import subprocess
import tempfile
import time
import pandas as pd
import anthropic
import google.generativeai as genai
import ast 
import re

LOG_FILENAME = "llm_benchmark_run_stdout.log"

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8') 

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)      

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if self.log and not self.log.closed:
            self.log.close()

# --- Configuration ---
CSV_INPUT_FILENAME = "code_contests_levels1to5.csv"
EVAL_RESULTS_FILENAME = "llm_scheme_evaluation_results.csv"
MAX_ATTEMPTS_PER_PROBLEM_LLM = 10
GUILE_TIMEOUT_SECONDS = 15

# --- API Setup ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY_CLAUDE environment variable not set. Claude evaluation will be skipped.")
if not GEMINI_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. Gemini evaluation will be skipped.")

anthropic_client = None
if ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"Could not initialize Claude client: {e}")

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
    except Exception as e:
         print(f"Warning: Could not initialize Gemini client. Gemini evaluation will be skipped. Error: {e}")
         gemini_model = None

MODELS_CONFIG = {
    "claude-4-opus-thinking": {
        "api_function": "call_claude_with_thinking",
        "model_name_actual": "claude-opus-4-20250514",
        "thinking_budget": 4096,
        "max_tokens_overall": 16384
    },
    "gemini-2.5-pro-thinking": { 
        "api_function": "call_gemini_with_thinking",
        "model_name_actual": "gemini-2.5-pro-preview-05-06", 
        "thinking_budget": 10000,
        "max_output_tokens": 25000
    }
}

def call_claude_with_thinking(prompt, model_name_actual, thinking_budget, max_tokens_overall):
    if not anthropic_client:
        print("      WARNING: Claude client not initialized for call_claude_with_thinking.")
        return None, "Claude client not initialized."
    try:
        print(f"      DEBUG: Calling Claude ({model_name_actual}) with thinking (budget: {thinking_budget}, max_tokens: {max_tokens_overall})")

        full_assistant_response_text = "" 
        thinking_log = [] 

        with anthropic_client.messages.stream(
            model=model_name_actual,
            max_tokens=max_tokens_overall,
            messages=[{"role": "user", "content": prompt}],
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        ) as stream:
            current_block_type = None 

            for event in stream:
                #print(f"      DEBUG: Claude Stream Event: type={event.type}") # Verbose: log every event type
                if event.type == "content_block_start":
                    current_block_type = event.content_block.type
                    if current_block_type == "thinking":
                        thinking_log.append(f"-- THINKING BLOCK {len(thinking_log)+1} START --")

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        if current_block_type == "text":
                            full_assistant_response_text += event.delta.text
                        elif current_block_type == "thinking": 
                            if thinking_log and isinstance(thinking_log[-1], str) and not thinking_log[-1].endswith("END --"):
                                if thinking_log[-1].startswith("-- THINKING BLOCK") and thinking_log[-1].endswith("START --"):
                                     thinking_log.append(event.delta.text) 
                                else:
                                     thinking_log[-1] += event.delta.text 
                            else:
                                thinking_log.append(event.delta.text) 
                
                elif event.type == "content_block_stop":
                    if current_block_type == "thinking":
                        if thinking_log and isinstance(thinking_log[-1], str) and not thinking_log[-1].startswith("-- THINKING BLOCK"):
                            thinking_log[-1] += "\n-- THINKING BLOCK END --"
                        elif thinking_log and isinstance(thinking_log[-1], str) and thinking_log[-1].startswith("-- THINKING BLOCK") and thinking_log[-1].endswith("START --"):
                            thinking_log[-1] = thinking_log[-1].replace("START --", "ENDED (empty) --")
                        else: 
                            thinking_log.append("-- THINKING BLOCK END (no start detected prior) --")
                    # print(f"        DEBUG: Stopped content_block: {current_block_type}")
                    current_block_type = None # Reset for the next block

                elif event.type == "message_stop":
                    # The stream is complete.
                    print(f"      DEBUG: Claude message stream finished. Reason: {event.message.stop_reason if hasattr(event, 'message') and hasattr(event.message, 'stop_reason') else 'N/A'}")
                    if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                         print(f"        DEBUG: Claude final usage: Input Tokens: {event.message.usage.input_tokens}, Output Tokens: {event.message.usage.output_tokens}, Thinking Tokens: {event.message.usage.thinking_tokens if hasattr(event.message.usage, 'thinking_tokens') else 'N/A'}")


        if thinking_log:
            print("      DEBUG: Claude thinking process captured:")
            for item in thinking_log:
                print(f"        {item[:300]}{'...' if len(item)>300 else ''}") # Print snippets of thinking log

        if not full_assistant_response_text.strip():
            print("      WARNING: Assistant response text from Claude stream is empty after processing.")
            if thinking_log:
                print("      DEBUG: Attempting to extract code from the last thinking block as a fallback...")
                potential_code_from_thinking = ""
                for item in reversed(thinking_log):
                    if isinstance(item, str) and not item.startswith("-- THINKING"): 
                        potential_code_from_thinking = item
                        break
                if potential_code_from_thinking:
                    print("      DEBUG: Found potential code in thinking log. Using it.")
                    return extract_scheme_from_llm_output(potential_code_from_thinking), None


        return extract_scheme_from_llm_output(full_assistant_response_text), None

    except anthropic.APIStatusError as e: 
        print(f"      ERROR: Anthropic APIStatusError ({model_name_actual} with thinking): {e.status_code} - {e.message}")
        import traceback
        traceback.print_exc()
        return None, f"Anthropic APIStatusError: {e.status_code} - {e.message}"
    except Exception as e:
        print(f"      ERROR: Claude API error ({model_name_actual} with thinking): {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return None, f"Claude API error: {type(e).__name__} - {e}"
    
def call_gemini_with_thinking(prompt, model_name_actual, thinking_budget, max_output_tokens_config):
    if not gemini_model:
        print("      WARNING: Gemini model not initialized for call_gemini_with_thinking.")
        return None, "Gemini model not initialized."
    try:
        active_gemini_model = genai.GenerativeModel(model_name_actual)

        generation_config_params = {
            "max_output_tokens": max_output_tokens_config,
            "temperature": 0.1,  
        }
        generation_config = genai.types.GenerationConfig(**generation_config_params)

        print(f"      DEBUG: Calling Gemini ({model_name_actual}) with max_tokens: {max_output_tokens_config}")
        print(f"      DEBUG: Note - Gemini doesn't have explicit thinking_config, relying on model's internal reasoning")

        enhanced_prompt = f"""Please think through this problem step by step before providing your solution.

{prompt}

Please show your reasoning process and then provide the final Scheme code."""
        
        response = active_gemini_model.generate_content(
            enhanced_prompt,
            generation_config=generation_config
        )

        if not response.candidates:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"      ERROR: Gemini prompt blocked ({model_name_actual}): {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}")
                return None, f"Gemini prompt blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}"
            print(f"      ERROR: Gemini response ({model_name_actual}) has no candidates.")
            return None, "Gemini response has no candidates."

        candidate = response.candidates[0]
        if candidate.finish_reason.name not in ['STOP', 'MAX_TOKENS']:
            print(f"      ERROR: Gemini generation ({model_name_actual}) finished due to: {candidate.finish_reason.name}")
            return None, f"Gemini generation finished due to: {candidate.finish_reason.name}"
        
        if candidate.content and candidate.content.parts:
            full_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            print(f"      DEBUG: Gemini response received ({len(full_text)} chars)")
            return extract_scheme_from_llm_output(full_text), None
        elif hasattr(response, 'text'): # Fallback if structure is simpler
            return extract_scheme_from_llm_output(response.text), None
        
        print(f"      ERROR: Gemini response text ({model_name_actual}) not found.")
        return None, "Gemini response text not found."
    
    except Exception as e: 
        print(f"      ERROR: Unhandled exception in call_gemini_with_thinking ({model_name_actual}): {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for this unhandled exception
        return None, f"Unhandled exception in call_gemini_with_thinking: {e}"


def extract_scheme_from_llm_output(text_response: str) -> str:
    """Extracts Scheme code from LLM response, handling markdown."""
    if not text_response:
        return ""
    scheme_block_start = text_response.find("```scheme")
    if scheme_block_start != -1:
        code_start = scheme_block_start + len("```scheme")
        scheme_block_end = text_response.find("```", code_start)
        if scheme_block_end != -1:
            return text_response[code_start:scheme_block_end].strip()
        else: 
            return text_response[code_start:].strip()

    lisp_block_start = text_response.find("```lisp")
    if lisp_block_start != -1:
        code_start = lisp_block_start + len("```lisp")
        lisp_block_end = text_response.find("```", code_start)
        if lisp_block_end != -1:
            return text_response[code_start:lisp_block_end].strip()
        else: # No closing ```
            return text_response[code_start:].strip()

    if text_response.strip().startswith(("(", "(define", "(lambda", ";")):
        return text_response.strip()


    generic_block_start = text_response.find("```")
    if generic_block_start != -1:
        first_line_end = text_response.find("\n", generic_block_start)
        if first_line_end == -1: first_line_end = len(text_response)
        first_line_tag = text_response[generic_block_start + 3 : first_line_end].strip().lower()

        code_actual_start = generic_block_start + 3 # Default start after ```
        if first_line_tag == "scheme" or first_line_tag == "lisp":
            # If tag is present, code starts after the newline following the tag
            nl_after_tag = text_response.find("\n", generic_block_start)
            if nl_after_tag != -1:
                code_actual_start = nl_after_tag + 1

        generic_block_end = text_response.find("```", code_actual_start)
        if generic_block_end != -1:
            return text_response[code_actual_start:generic_block_end].strip()
        else: # No closing ```, assume rest of string from code_actual_start is code
            return text_response[code_actual_start:].strip()

    return text_response.strip()


def call_claude(prompt):
    if not anthropic_client:
        return None, "Claude client not initialized."
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        if message.content and message.content[0].type == 'text':
            return extract_scheme_from_llm_output(message.content[0].text), None
        else:
            return None, "Claude response did not contain text content."
    except Exception as e:
        return None, f"Claude API error: {e}"

def call_gemini(prompt):
    if not gemini_model:
        return None, "Gemini model not initialized."
    try:
        response = gemini_model.generate_content(prompt)
        if not response.candidates:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return None, f"Gemini prompt blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}"
            return None, "Gemini response has no candidates."

        candidate = response.candidates[0]
        if candidate.finish_reason.name not in ['STOP', 'MAX_TOKENS']:
            return None, f"Gemini generation finished due to: {candidate.finish_reason.name}"

        if candidate.content and candidate.content.parts:
            full_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            return extract_scheme_from_llm_output(full_text), None
        elif hasattr(response, 'text'): # Fallback
             return extract_scheme_from_llm_output(response.text), None
        return None, "Gemini response text not found."
    except Exception as e:
        return None, f"Gemini API error: {e}"

def to_msys_path(windows_path):
    """Converts a Windows path to an MSYS-like path (e.g., C:\foo -> /c/foo)."""
    if os.name != 'nt': # Only convert on Windows
        return windows_path
    
    path = windows_path.replace('\\', '/')
    if ':' in path:
        drive, rest_of_path = path.split(':', 1)
        path = f"/{drive.lower()}{rest_of_path}"
    return path

def run_scheme_code(code, public_tests_input_list, timeout=10):
    scheme_code_as_str = "" # Initialize
    if isinstance(code, bytes):
        try:
            scheme_code_as_str = code.decode('utf-8')
        except UnicodeDecodeError:
            print("      DEBUG: 'code' was bytes and failed UTF-8 decode. Trying 'latin-1'.")
            scheme_code_as_str = code.decode('latin-1', errors='replace')
    elif code is not None:
        scheme_code_as_str = str(code)
    else:
        scheme_code_as_str = ""

    if not scheme_code_as_str.strip():
        return False, None, "No actual Scheme code to write after processing."

    tmp_file_path = None
    try:
        print(f"      DEBUG: Preparing to write. Type of scheme_code_as_str: {type(scheme_code_as_str)}")
        
        encoded_scheme_code_for_file = None
        try:
            encoded_scheme_code_for_file = scheme_code_as_str.encode('utf-8')
            print(f"      DEBUG: Successfully encoded scheme_code_as_str to UTF-8 bytes. Length: {len(encoded_scheme_code_for_file)}")
        except UnicodeEncodeError as uee:
            print(f"      DEBUG: FAILED to encode scheme_code_as_str to UTF-8: {uee}")
            print(f"      DEBUG: Offending character: {scheme_code_as_str[uee.start:uee.end]}")
            print(f"      DEBUG: Trying to encode with 'utf-8' and 'replace' errors.")
            try:
                encoded_scheme_code_for_file = scheme_code_as_str.encode('utf-8', errors='replace')
                print(f"      DEBUG: Successfully encoded with 'utf-8' and 'replace'. Length: {len(encoded_scheme_code_for_file)}")
            except Exception as e_replace:
                print(f"      DEBUG: FAILED to encode even with 'utf-8' and 'replace': {e_replace}")
                return False, None, f"Failed to prepare code for file due to encoding: {uee}"
        
        if encoded_scheme_code_for_file is None:
            return False, None, "Code could not be encoded for file writing."

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.scm', delete=False) as tmp_file: 
            tmp_file_path = tmp_file.name
            try:
                print(f"      DEBUG: Writing {len(encoded_scheme_code_for_file)} bytes to {tmp_file_path}")
                tmp_file.write(encoded_scheme_code_for_file) # Write bytes
                print(f"      DEBUG: Successfully wrote bytes to temporary file.")
            except Exception as write_err:
                print(f"      DEBUG: ERROR during tmp_file.write(encoded_bytes): {write_err}") 
                raise 

        actual_outputs_str = ""
        error_message = None
        execution_success = True

        input_string = ""
        for single_test_input in public_tests_input_list:
            input_string += str(single_test_input)
            if not str(single_test_input).endswith('\n'):
                 input_string += "\n"

        tmp_file_path_windows = tmp_file.name
        temp_dir = os.path.dirname(tmp_file_path_windows)
        temp_filename = os.path.basename(tmp_file_path_windows)

        print(f"      DEBUG: Executing Guile: ['guile', '-s', '{temp_filename}'] in cwd: {temp_dir}")
        process = subprocess.run(
            ['guile', '-s', temp_filename], 
            input=input_string.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=temp_dir 
        )

        stdout_bytes = process.stdout
        stderr_bytes = process.stderr
        
        try:
            actual_outputs_str = stdout_bytes.decode('utf-8')
        except UnicodeDecodeError:
            print("      DEBUG: stdout was not valid UTF-8, trying latin-1 with replace")
            actual_outputs_str = stdout_bytes.decode('latin-1', errors='replace')
        
        stderr_str = ""
        try:
            stderr_str = stderr_bytes.decode('utf-8')
        except UnicodeDecodeError:
            print("      DEBUG: stderr was not valid UTF-8, trying latin-1 with replace")
            stderr_str = stderr_bytes.decode('latin-1', errors='replace')
        
        print(f"      DEBUG: Guile process RC: {process.returncode}")
        if len(stderr_str.strip()) > 0 :
            print(f"      DEBUG: Guile stderr (first 200 chars): {stderr_str.strip()[:200]}")
        if len(actual_outputs_str.strip()) > 0:
            print(f"      DEBUG: Guile stdout (first 200 chars): {actual_outputs_str.strip()[:200]}")


        if process.returncode != 0:
            execution_success = False
            error_message = stderr_str.strip() if stderr_str.strip() else f"Execution failed with return code {process.returncode}."
        elif stderr_str.strip() and "WARNING" not in stderr_str.upper() and "NOTE" not in stderr_str.upper():
            pass 

    except subprocess.TimeoutExpired:
        execution_success = False
        error_message = "TimeoutOccurred"
    except FileNotFoundError:
        execution_success = False
        error_message = "Guile executable not found."
    except TypeError as te: 
        execution_success = False
        error_message = f"TypeError caught in run_scheme_code: {te}"
        print(f"      DEBUG: Caught Outer TypeError: {te}")
    except Exception as e:
        execution_success = False
        error_message = f"An unexpected error occurred during execution: {e}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return execution_success, actual_outputs_str if execution_success else None, error_message


def evaluate_test_cases(actual_output_str, expected_outputs_list):
    if actual_output_str is None or not expected_outputs_list:
        return False
    expected_full_output_str = "\n".join(map(str, expected_outputs_list)).strip()
    actual_normalized_str = actual_output_str.strip()
    return actual_normalized_str == expected_full_output_str


def parse_test_data_string(test_str):
    """
    Parses strings like "{'input': array(['...'], dtype=object), 'output': array(['...'], dtype=object)}"
    into a dictionary {'input': ['...'], 'output': ['...']}.
    Handles cases where input/output might be empty or malformed.
    """
    default_empty = {'input': [], 'output': []}
    if pd.isna(test_str) or not isinstance(test_str, str) or test_str.lower() == 'nan':
        return default_empty

    try:
        if not "array(" in test_str:
            evaluated = ast.literal_eval(test_str)
            if isinstance(evaluated, dict) and 'input' in evaluated and 'output' in evaluated:
                evaluated['input'] = [str(item) for item in evaluated.get('input', [])]
                evaluated['output'] = [str(item) for item in evaluated.get('output', [])]
                return evaluated
            else:
                return default_empty 
    except (ValueError, SyntaxError, TypeError):
        pass 

    parsed_data = {}
    for key in ['input', 'output']:
        match = re.search(rf"'{key}':\s*array\(\s*(\[.*?\])\s*,\s*dtype=object\)", test_str, re.DOTALL)
        if match:
            list_str = match.group(1)
            try:
                parsed_list = ast.literal_eval(list_str)
                parsed_data[key] = [str(item) for item in parsed_list if isinstance(item, (str, bytes, int, float))]
            except (SyntaxError, ValueError):
                # print(f"Warning: Could not parse list string for key '{key}': {list_str}")
                parsed_data[key] = []
        else:
            # print(f"Warning: Could not find array data for key '{key}' in '{test_str[:100]}...'")
            parsed_data[key] = []
            simple_match = re.search(rf"'{key}':\s*(\[.*?\])", test_str, re.DOTALL)
            if simple_match:
                list_str = simple_match.group(1)
                try:
                    parsed_list = ast.literal_eval(list_str)
                    parsed_data[key] = [str(item) for item in parsed_list if isinstance(item, (str, bytes, int, float))]
                except (SyntaxError, ValueError):
                    parsed_data[key] = [] # Keep it empty if simple list parse also fails

    if 'input' not in parsed_data: parsed_data['input'] = []
    if 'output' not in parsed_data: parsed_data['output'] = []

    return parsed_data

# --- Main Evaluation Loop ---
def main():
    stdout_logger = Logger(LOG_FILENAME)
    sys.stdout = stdout_logger
    
    stderr_logger = Logger("llm_benchmark_stderr.log")
    sys.stderr = stderr_logger

    print(f"Script started. Console output logged to {LOG_FILENAME} and llm_benchmark_stderr.log")

    print(f"Loading ALL problem data from {CSV_INPUT_FILENAME}...")
    if not os.path.exists(CSV_INPUT_FILENAME):
        print(f"CRITICAL ERROR: Input problem data file {CSV_INPUT_FILENAME} not found. Cannot proceed.")
        return
    try:
        df_problems_source = pd.read_csv(CSV_INPUT_FILENAME)
        if 'public_tests' in df_problems_source.columns:
            df_problems_source['public_tests_parsed'] = df_problems_source['public_tests'].apply(parse_test_data_string)
        else:
            print(f"WARNING: 'public_tests' column not found in {CSV_INPUT_FILENAME}. Tests might be empty for problems.")
            df_problems_source['public_tests_parsed'] = pd.Series([{'input': [], 'output': []}] * len(df_problems_source))
        
        all_problems_to_evaluate = []
        for i, row in df_problems_source.iterrows():
            all_problems_to_evaluate.append({
                "name": row.get('name', f"Problem_{i}_from_source"), 
                "description": str(row.get('description', '')),
                "public_tests_parsed": row.get('public_tests_parsed', {'input': [], 'output': []})
            })
        print(f"Loaded {len(all_problems_to_evaluate)} problems to evaluate from {CSV_INPUT_FILENAME}.")
    except Exception as e:
        print(f"CRITICAL ERROR loading or parsing {CSV_INPUT_FILENAME}: {e}")
        import traceback
        traceback.print_exc()
        return

    existing_eval_results_list = []
    already_processed_problem_llm_pairs = set() # To store (problem_name, llm_name) tuples

    if os.path.exists(EVAL_RESULTS_FILENAME):
        print(f"Loading existing evaluation results from {EVAL_RESULTS_FILENAME}...")
        try:
            df_existing_results = pd.read_csv(EVAL_RESULTS_FILENAME)
            existing_eval_results_list = df_existing_results.to_dict('records')
            for res in existing_eval_results_list:
                if res.get('problem_name') and res.get('llm_name'):
                    already_processed_problem_llm_pairs.add((res['problem_name'], res['llm_name']))
            print(f"Loaded {len(existing_eval_results_list)} existing results. Found {len(already_processed_problem_llm_pairs)} already processed (problem, llm) pairs.")
        except Exception as e:
            print(f"WARNING: Error loading {EVAL_RESULTS_FILENAME}: {e}. Will proceed as if it's a new file for appending purposes, but existing data might be lost if saving overwrites.")
            existing_eval_results_list = [] 
            already_processed_problem_llm_pairs = set()
    else:
        print(f"{EVAL_RESULTS_FILENAME} not found. Will create a new results file.")

    new_results_this_run = []
    problems_processed_count_this_run = 0 

    
    llms_to_run_config = {}
    llms_to_run_config.update(MODELS_CONFIG)

    if not llms_to_run_config:
        print("CRITICAL ERROR: No LLMs configured in llms_to_run_config. Exiting.")
        return

    print(f"\nWill attempt to evaluate problems with these LLM configurations: {list(llms_to_run_config.keys())}")

    for problem_idx, problem_data_dict in enumerate(all_problems_to_evaluate):
        problem_name = problem_data_dict['name']
        problem_description = problem_data_dict['description']
        public_tests_dict = problem_data_dict['public_tests_parsed']

        print(f"\n--- Processing Problem {problem_idx + 1}/{len(all_problems_to_evaluate)}: {problem_name} ---")

        current_problem_test_inputs = public_tests_dict.get('input', [])
        current_problem_test_outputs = public_tests_dict.get('output', [])

        if not (isinstance(public_tests_dict, dict) and \
                current_problem_test_inputs and \
                current_problem_test_outputs and \
                len(current_problem_test_inputs) == len(current_problem_test_outputs)):
            print(f"  WARNING: Skipping evaluation of {problem_name} due to missing/malformed public tests in source data.")
            new_results_this_run.append({
                'problem_name': problem_name, 'llm_name': 'N/A_SKIPPED_BAD_TESTS', 
                'solved': False, 'attempts_used': 0, 'final_code': '',
                'execution_error_details': 'Skipped due to bad tests in source CSV',
                'test_case_match_on_solve_or_last_attempt': False
            })
            continue

        for llm_id_key, config in llms_to_run_config.items():

            if (problem_name, llm_id_key) in already_processed_problem_llm_pairs:
                print(f"  INFO: Skipping {problem_name} with {llm_id_key} as it's already in existing results file.")
                continue

            print(f"\n  Evaluating {problem_name} with {llm_id_key} (Model: {config['model_name_actual']})...")

            api_func = globals()[config["api_function"]] # Get the function object

            solved = False
            attempts_used = 0
            final_code = ""
            execution_error_details = "No attempts made or all failed before execution."
            test_case_match_final = False

            for attempt_num in range(MAX_ATTEMPTS_PER_PROBLEM_LLM):
                attempts_used = attempt_num + 1
                print(f"    Attempt {attempts_used}/{MAX_ATTEMPTS_PER_PROBLEM_LLM}...")
                
                prompt = f"""Problem Name: {problem_name} 
Problem Description:
{problem_description}
Public Test Input Example (first one, content may contain newlines):
---BEGIN INPUT EXAMPLE---
{current_problem_test_inputs[0] if current_problem_test_inputs else "N/A"}
---END INPUT EXAMPLE---
Public Test Output Example (first one, content may contain newlines):
---BEGIN OUTPUT EXAMPLE---
{current_problem_test_outputs[0] if current_problem_test_outputs else "N/A"}
---END OUTPUT EXAMPLE---

Your task is to write a Scheme program for the Guile compiler that solves this problem.
Think step-by-step if possible to arrive at the solution. Ensure your program reads ALL input from standard input (stdin)
and prints ALL output to standard output (stdout), exactly matching the problem's requirements.
If the problem specifies a number of test cases T within the input, your program must loop T times.
Provide only the Scheme code itself. If using markdown, use ```scheme ... ```.

Scheme code:
"""
                if "claude" in config["api_function"].lower():
                    code, api_error = api_func(prompt, 
                                            config["model_name_actual"], 
                                            config.get("thinking_budget"), 
                                            config.get("max_tokens_overall"))
                elif "gemini" in config["api_function"].lower():
                    code, api_error = api_func(prompt, 
                                            config["model_name_actual"], 
                                            config.get("thinking_budget"), 
                                            config.get("max_output_tokens"))
                else:
                    print(f"      ERROR: Unknown API function type for {config['api_function']}")
                    execution_error_details = "Unknown API function type"
                    break 

                time.sleep(5) 

                if api_error:
                    print(f"      API Error ({llm_id_key}): {api_error}")
                    execution_error_details = f"Attempt {attempts_used} API Error: {api_error}"
                    final_code = f"API_ERROR: {api_error}"
                    if any(s in str(api_error).lower() for s in ["rate limit", "quota", "usage limit"]):
                        print("      WARNING: Rate limit / Quota error encountered. Sleeping for 30s.")
                        time.sleep(30)
                    continue 
                if not code or not code.strip():
                    print(f"      WARNING: LLM ({llm_id_key}) returned empty code.")
                    execution_error_details = f"Attempt {attempts_used} LLM returned empty code."
                    final_code = "EMPTY_CODE_RETURNED"
                    continue

                final_code = code
                print(f"      Generated code ({len(code)} chars) by {llm_id_key}. Preview: {code[:100].replace('\n', '\\n')}...")

                exec_ok, actual_stdout, run_err = run_scheme_code(code, current_problem_test_inputs, timeout=GUILE_TIMEOUT_SECONDS)

                if not exec_ok:
                    execution_error_details = f"Attempt {attempts_used} Execution Error ({llm_id_key}): {run_err}"
                    print(f"      WARNING: {str(run_err)}") 
                    test_case_match_final = False
                    continue

                test_case_match_final = evaluate_test_cases(actual_stdout, current_problem_test_outputs)

                if test_case_match_final:
                    print(f"      Test cases PASSED for {problem_name} with {llm_id_key}!")
                    solved = True
                    execution_error_details = f"Solved by {llm_id_key}"
                    break 
                else:
                    execution_error_details = f"Attempt {attempts_used} Test Cases FAILED ({llm_id_key}, Output Mismatch)."
                    print(f"      WARNING: {execution_error_details}")
            current_run_result = {
                'problem_name': problem_name,
                'llm_name': llm_id_key, 
                'solved': solved,
                'attempts_used': attempts_used,
                'final_code': final_code,
                'execution_error_details': execution_error_details,
                'test_case_match_on_solve_or_last_attempt': test_case_match_final
            }
            new_results_this_run.append(current_run_result)
            already_processed_problem_llm_pairs.add((problem_name, llm_id_key)) 

        problems_processed_count_this_run += 1

        if problems_processed_count_this_run % 5 == 0 or problem_idx == len(all_problems_to_evaluate) - 1:
            if new_results_this_run: 
                print(f"\n--- Saving results incrementally ({problems_processed_count_this_run} problems processed in this run batch) ---")
                
                combined_for_saving = existing_eval_results_list + new_results_this_run 
                
                try:
                    df_to_save = pd.DataFrame(combined_for_saving)
                    df_to_save.to_csv(EVAL_RESULTS_FILENAME, index=False)
                    print(f"Results (total {len(df_to_save)}) saved successfully to {EVAL_RESULTS_FILENAME}")
                except Exception as e:
                    print(f"ERROR saving results to CSV during incremental save: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n--- No new results to save incrementally at this point ({problems_processed_count_this_run} problems processed) ---")
    
    if new_results_this_run: 
        print(f"\n--- Performing final save of all results ---")
        final_combined_results = existing_eval_results_list + new_results_this_run
        try:
            df_final_save = pd.DataFrame(final_combined_results)
            df_final_save.drop_duplicates(subset=['problem_name', 'llm_name'], keep='last', inplace=True)
            df_final_save.to_csv(EVAL_RESULTS_FILENAME, index=False)
            print(f"Final results (total {len(df_final_save)}) saved successfully to {EVAL_RESULTS_FILENAME}")
        except Exception as e:
            print(f"ERROR during final save of results to CSV: {e}")
            import traceback
            traceback.print_exc()
    elif not existing_eval_results_list and not new_results_this_run : 
        print("No evaluation results were collected or generated to save.")
    elif existing_eval_results_list and not new_results_this_run: 
        print("No new results generated in this run. Existing results file remains unchanged.")


    print("Script finished.")

if __name__ == "__main__":
    try:
        import numpy
    except ImportError:
        print("Warning: numpy not installed. Some parsing features might be limited if 'array(...)' appears in unexpected ways.")

    main()