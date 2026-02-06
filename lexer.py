import ply.lex as lex
import sys
import os

# --- A New, Centralized Map for Token IDs ---
# This maps the PLY token TYPE (string) to the required token ID (integer)
TOKEN_ID_MAP = {
    # Types and Literals
    'TYPE': 301,
    'CHARACTER': 302,
    'INTEGER': 303,
    'REAL': 304,
    'STRING': 305,
    'IDENTIFIER': 306,
    'HEXADECIMAL': 307,
    'BOOLEAN_OP': 308, # For ^ and ^=
    
    # Two-character operators
    'EQUAL': 351, 'NEQUAL': 352, 'GEQUAL': 353, 'LEQUAL': 354, 'INCREMENT': 355,
    'DECREMENT': 356, # Note: PDF typo, should be 356. Using 356 for --
    'OR': 357, # Note: PDF typo, should be 357 for ||.
    'AND': 358, 'ADD_ASSIGN': 361, 'SUB_ASSIGN': 362, 'MUL_ASSIGN': 363, 'DIV_ASSIGN': 364,
    'ARROW': 365,

    # Keywords (PLY will generate types like 'IF', 'WHILE', etc.)
    'CONST': 401, 'STRUCT': 402, 'FOR': 403, 'WHILE': 404, 'DO': 405, 'IF': 406,
    'ELSE': 407, 'BREAK': 408, 'CONTINUE': 409, 'RETURN': 410, 'SWITCH': 411,
    'CASE': 412, 'DEFAULT': 413, 'TRUE': 414, 'FALSE': 415, 'BOOL': 416
}

# Dictionary of reserved keywords for PLY
reserved = {
    'const': 'CONST', 'struct': 'STRUCT', 'void': 'TYPE', 'char': 'TYPE',
    'int': 'TYPE', 'float': 'TYPE', 'for': 'FOR', 'while': 'WHILE', 'do': 'DO',
    'if': 'IF', 'else': 'ELSE', 'break': 'BREAK', 'continue': 'CONTINUE',
    'return': 'RETURN', 'switch': 'SWITCH', 'case': 'CASE', 'default': 'DEFAULT',
    'true': 'TRUE', 'false': 'FALSE', 'bool': 'BOOL'
}

# Base list of tokens that are not keywords
base_tokens = [
   'IDENTIFIER', 'INTEGER', 'REAL', 'STRING', 'CHARACTER', 'HEXADECIMAL', 'ARROW',
   # Two-char operators
   'INCREMENT', 'DECREMENT', 'ADD_ASSIGN', 'SUB_ASSIGN', 'MUL_ASSIGN', 'DIV_ASSIGN',
   'EQUAL', 'NEQUAL', 'LEQUAL', 'GEQUAL', 'AND', 'OR', 'XOR_ASSIGN',
   # Single-char symbols (we will handle their IDs separately)
   'PLUS', 'MINUS', 'STAR', 'SLASH', 'ASSIGN', 'LESS', 'GREATER', 'LPAREN',
   'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'SEMICOLON', 'COMMA',
   'DOT', 'BITWISE_AND', 'BITWISE_OR', 'NOT', 'QUESTION_MARK', 'MODULO', 'COLON',
   'XOR' # Added for the ^ operator
]
# Use a set to get unique token names from the reserved map, then combine lists.
tokens = base_tokens + list(set(reserved.values()))


# Define length limits
MAX_INTEGER_LENGTH = 48
MAX_REAL_LENGTH = 48
MAX_IDENTIFIER_LENGTH = 48
MAX_STRING_LENGTH = 1024

# --- PLY Token Definitions ---

# Whitespace (tabs and spaces)
t_ignore = ' \t'

# An error flag to be used by the main function
error_found = False


def t_ignore_COMMENT(t):
    r'/\*[\s\S]*?\*/|//.*'
    t.lexer.lineno += t.value.count('\n')
    pass # Discard the token by returning nothing.
def t_INVALID_IDENTIFIER(t):
    r'\d+[a-zA-Z_][a-zA-Z0-9_]*'
    custom_error(t, f"Invalid identifier format")
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER') # Check for reserved words
    if len(t.value) > MAX_IDENTIFIER_LENGTH:
        custom_error(t, f"Identifier '{t.value[:20]}...' is too long")
    return t

def t_HEXADECIMAL(t):
    r'0[xX][0-9a-fA-F]+'
    # Convert the hex string (e.g., "0x1A") to its integer decimal equivalent (26)
    # and store it back as a string.
    t.value = str(int(t.value, 16))
    return t

def t_REAL(t):
    r'(\d+\.\d*|\.\d+)([eE][+-]?\d+)?|\d+[eE][+-]?\d+'
    if len(t.value) > MAX_REAL_LENGTH:
        custom_error(t, f"Real literal '{t.value[:20]}...' is too long")
    return t

def t_INTEGER(t):
    r'\d+'
    if len(t.value) > MAX_INTEGER_LENGTH:
        custom_error(t, f"Integer literal '{t.value[:20]}...' is too long")
    return t


def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    if len(t.value[1:-1]) > MAX_STRING_LENGTH:
        custom_error(t, f"String literal is too long")
    # Handle newline characters that might be part of the raw string
    t.lexer.lineno += t.value.count('\n')
    # The token's value is already the full matched string, which is what's desired.
    return t

def t_CHARACTER(t):
    r"'([^'\\]|\\.)'"
    # The token's value is the raw character literal, e.g., "'a'", which is what's desired.
    return t

# Two-character operators must be defined before single-character ones
t_INCREMENT   = r'\+\+'
t_DECREMENT   = r'--'
t_ADD_ASSIGN  = r'\+='
t_SUB_ASSIGN  = r'-='
t_MUL_ASSIGN  = r'\*='
t_DIV_ASSIGN  = r'/='
t_EQUAL       = r'=='
t_NEQUAL      = r'!='
t_LEQUAL      = r'<='
t_GEQUAL      = r'>='
t_AND         = r'&&'
t_OR          = r'\|\|'
t_ARROW       = r'->'
t_XOR_ASSIGN  = r'\^=' # Added for Boolean Operation

# Single-character symbols
t_PLUS        = r'\+'
t_MINUS       = r'-'
t_STAR        = r'\*'
t_SLASH       = r'/'
t_ASSIGN      = r'='
t_LESS        = r'<'
t_GREATER     = r'>'
t_LPAREN      = r'\('
t_RPAREN      = r'\)'
t_LBRACE      = r'\{'
t_RBRACE      = r'\}'
t_LBRACKET    = r'\['
t_RBRACKET    = r'\]'
t_SEMICOLON   = r';'
t_COMMA       = r','
t_DOT         = r'\.'
t_BITWISE_AND = r'&'
t_BITWISE_OR  = r'\|'
t_NOT         = r'!'
t_QUESTION_MARK = r'\?'
t_MODULO      = r'%'
t_COLON       = r':'
t_XOR         = r'\^' 


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_error(t):
    """Handles invalid characters and malformed literals."""
    lexer = t.lexer
    # If the error starts with a single quote, handle it as a single block.
    if t.value[0] == "'":
        # Scan forward to find the logical end of the error token
        # It ends at the next quote, or the end of the line, whichever is first.
        next_quote_pos = lexer.lexdata.find("'", lexer.lexpos + 1)
        newline_pos = lexer.lexdata.find("\n", lexer.lexpos + 1)

        end_pos = -1
        if next_quote_pos != -1 and (newline_pos == -1 or next_quote_pos < newline_pos):
            end_pos = next_quote_pos + 1
        elif newline_pos != -1:
            end_pos = newline_pos
        else: # Reaches end of file
            end_pos = len(lexer.lexdata)
        
        malformed_text = lexer.lexdata[lexer.lexpos:end_pos]
        t.value = malformed_text # Update the token's value for the error message
        custom_error(t, f"Malformed or unclosed character literal")
        lexer.skip(len(malformed_text))
    
    elif t.value[0] == '"':
        # An unclosed string runs to the end of the line or file.
        newline_pos = lexer.lexdata.find("\n", lexer.lexpos + 1)
        end_pos = newline_pos if newline_pos != -1 else len(lexer.lexdata)
        
        malformed_text = lexer.lexdata[lexer.lexpos:end_pos]
        t.value = malformed_text
        custom_error(t, f"Unclosed string literal")
        lexer.skip(len(malformed_text))
    else:
        
        message = f"Invalid character '{t.value[0]}'"
        custom_error(t, message)
        lexer.skip(1)

def custom_error(t, message):
    """A centralized function for reporting errors."""
    global error_found
    error_found = True
    sys.stderr.write(f"Lexer error in file {t.lexer.filename} line {t.lineno} at text {t.value}\n")
    sys.stderr.write(f"  Description: {message}\n")

# --- Main Lexer Function ---
def run_lexical_analysis(filename):
    global error_found
    error_found = False # Reset error flag for each run

    try:
        # Explicitly open the file with UTF-8 encoding.
        
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            source_code = f.read()
    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file not found at '{filename}'\n")
        sys.exit(1)

    # Re-map the new boolean operators to a single type for the ID map
    TOKEN_ID_MAP['XOR'] = TOKEN_ID_MAP['BOOLEAN_OP']
    TOKEN_ID_MAP['XOR_ASSIGN'] = TOKEN_ID_MAP['BOOLEAN_OP']

    lexer = lex.lex()
    lexer.filename = filename # Store filename for error messages
    lexer.input(source_code)

    output_filename = os.path.splitext(filename)[0] + ".lexer"
    
    # Process tokens and store them before writing to file
    
    output_lines = []
    while True:
        tok = lexer.token()
        if not tok:
            break # No more tokens

        # --- THIS IS THE NEW, SIMPLIFIED ID LOGIC ---
        token_id = -1 # Default
        if tok.type in TOKEN_ID_MAP:
            # If the type is in our map, use that ID
            token_id = TOKEN_ID_MAP[tok.type]
        elif len(tok.value) == 1 and tok.type not in ['IDENTIFIER', 'INTEGER', 'REAL', 'STRING', 'CHARACTER']:
            # Otherwise, if it's a single character symbol, use its ASCII value
            token_id = ord(tok.value)
        
        line = f"File {filename} Line {tok.lineno} Token {token_id} Text {tok.value}"
        output_lines.append(line)

    
    if not error_found:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_lines))
                f.write("\n")
            print(f"Successfully generated token file at: {output_filename}")
        except IOError:
            sys.stderr.write(f"Error: Could not write to output file '{output_filename}'.\n")
            sys.exit(1)
    else:
        # As per spec, if an error occurred, remove the output file if it exists
        print("Lexing failed due to errors.", file=sys.stderr)
        if os.path.exists(output_filename):
            os.remove(output_filename)

