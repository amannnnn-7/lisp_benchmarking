import sys
import re

def tokenize(chars):
    """
    Converts a string of Scheme code into a list of tokens.
    It adds spaces around parentheses to facilitate splitting.
    """
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

def parse_expression(tokens):
    """
    Parses a list of tokens into a single S-expression.
    This function is called recursively to handle nested expressions.
    """
    if not tokens:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens and tokens[0] != ')':
            L.append(parse_expression(tokens))
        if not tokens:
            raise SyntaxError("Unmatched '('")
        tokens.pop(0) # pop ')'
        return L
    elif token == ')':
        raise SyntaxError("Unexpected ')'")
    else:
        return token

def parse_scheme_code(code_string):
    """
    Parses an entire Scheme code string into a list of top-level S-expressions.
    """
    tokens = tokenize(code_string)
    expressions = []
    while tokens:
        expressions.append(parse_expression(tokens))
    return expressions

# --- Function Analysis ---

def get_user_defined_functions(parsed_code):
    """
    Identifies and extracts user-defined function definitions from the parsed Scheme code.
    Returns a dictionary mapping function names to their S-expression bodies.
    """
    functions_info = {} 
    for expr in parsed_code:
        if (isinstance(expr, list) and len(expr) > 2 and expr[0] == 'define' and
            isinstance(expr[1], list) and len(expr[1]) > 0):
            func_name = expr[1][0] 
            func_body = expr[2:]   
            functions_info[func_name] = func_body
    return functions_info

def find_all_calls_in_body(body_s_expressions, user_defined_func_names):
    """
    Traverses the S-expression body of a function to find all calls to
    other user-defined functions.
    """
    called_user_funcs = set()
    
    def traverse_and_collect(expr):
        if isinstance(expr, list):
            if len(expr) > 0:
                head = expr[0]
                if isinstance(head, str) and head in user_defined_func_names:
                    called_user_funcs.add(head)

                for sub_expr in expr:
                    traverse_and_collect(sub_expr)

    for expr in body_s_expressions:
        traverse_and_collect(expr)
        
    return called_user_funcs

def assign_function_levels(functions_info):
    """
    Assigns a dependency level to each user-defined function.
    """
    function_levels = {} 
    user_defined_func_names = set(functions_info.keys())
    
    dependencies = {} # {func_name: set_of_user_defined_dependencies}
    for func_name, body in functions_info.items():
        dependencies[func_name] = find_all_calls_in_body(body, user_defined_func_names)
    
    # Keep track of functions that still need a level assigned
    unassigned_functions = set(user_defined_func_names)
    
    while unassigned_functions:
        assigned_in_this_pass = set()
        
        for func_name in list(unassigned_functions):
            
            all_deps_assigned = True
            max_dep_level = 0 
            
            for dep_name in dependencies[func_name]:
                if dep_name not in function_levels:
                    all_deps_assigned = False
                    break
                max_dep_level = max(max_dep_level, function_levels[dep_name])
            
            if all_deps_assigned:
                function_levels[func_name] = max_dep_level + 1
                assigned_in_this_pass.add(func_name)
        
        if not assigned_in_this_pass:
            print("Warning: Could not assign levels to all functions. Possible circular dependencies.", file=sys.stderr)
            print(f"Unassigned functions: {unassigned_functions}", file=sys.stderr)
            break 

        unassigned_functions -= assigned_in_this_pass
        
    return function_levels


def main():
    """
    Main function to handle file input, parsing, level assignment, and output.
    """
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        print("Usage: python parse_scheme.py -- <input_file_path>", file=sys.stderr)
        sys.exit(1)
    
    input_file_path = sys.argv[2]

    try:
        with open(input_file_path, 'r') as f:
            scheme_code = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{input_file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse the Scheme code into S-expressions
    try:
        parsed_code = parse_scheme_code(scheme_code)
    except SyntaxError as e:
        print(f"Error parsing Scheme code: {e}", file=sys.stderr)
        sys.exit(1)
    
    functions_info = get_user_defined_functions(parsed_code)
    
    if not functions_info:
        print("No user-defined functions found in the provided Scheme code.", file=sys.stderr)
        return

    # Assign levels to the functions
    function_levels = assign_function_levels(functions_info)
    
    # Group functions by their assigned level for output
    levels_map = {}
    for func_name, level in function_levels.items():
        levels_map.setdefault(level, []).append(func_name)

    sorted_levels = sorted(levels_map.keys())
    
    for level in sorted_levels:
        functions_at_level = sorted(levels_map[level])
        print(f"Level {level}: {', '.join(functions_at_level)}")

if __name__ == "__main__":
    main()