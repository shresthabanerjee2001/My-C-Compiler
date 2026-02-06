import ply.yacc as yacc
import ply.lex as lex
import os
import sys

import lexer as clexer




class Node:
    def __init__(self, filename):
        self.filename = filename

class ProgramNode(Node):
    def __init__(self, declarations, filename):
        super().__init__(filename)
        self.declarations = declarations

class VarDeclNode(Node):
    def __init__(self, var_type, identifiers, lineno, filename):
        super().__init__(filename)
        self.var_type = var_type
        self.identifiers = identifiers
        self.lineno = lineno

class InitDeclaratorNode(Node):
    def __init__(self, identifier, expression, lineno, filename):
        super().__init__(filename)
        self.identifier = identifier
        self.expression = expression
        self.lineno = lineno # Line of the '='

class FuncDefNode(Node):
    def __init__(self, return_type, name_node, params, body, lineno, filename):
        super().__init__(filename)
        self.return_type = return_type
        self.name_node = name_node
        self.params = params
        self.body = body
        self.lineno = lineno 

class FuncPrototypeNode(Node):
    def __init__(self, return_type, name_node, params, lineno, filename):
        super().__init__(filename)
        self.return_type = return_type
        self.name_node = name_node
        self.params = params
        self.lineno = lineno # Line of function name

class ParamNode(Node):
    def __init__(self, param_type, ident_node, lineno, filename):
        super().__init__(filename)
        # param_type will be like "int" or "int[]"
        self.param_type = param_type
        self.ident_node = ident_node
        self.lineno = lineno

class IdentifierNode(Node):
    def __init__(self, name, lineno, filename, is_array=False):
        super().__init__(filename)
        self.name = name
        self.lineno = lineno
       
        self.is_array = is_array
        
class StructDefNode(Node):
    def __init__(self, name_node, members, lineno, filename):
        super().__init__(filename)
        self.name_node = name_node
        self.members = members
        self.lineno = lineno # Line of struct name

class MemberNode(Node):
    def __init__(self, member_type, ident_node, lineno, filename):
        super().__init__(filename)
        self.member_type = member_type
        self.ident_node = ident_node
        self.lineno = lineno

class StmtBlockNode(Node):
    def __init__(self, var_declarations, statements, lineno, filename):
        super().__init__(filename)
        self.var_declarations = var_declarations
        self.statements = statements
        self.lineno = lineno # Line of '{'

class IfStmtNode(Node):
    def __init__(self, condition, then_stmt, else_stmt, lineno, filename):
        super().__init__(filename)
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt
        self.lineno = lineno # Line of 'if'

class ForStmtNode(Node):
    def __init__(self, init, cond, update, body, lineno, filename):
        super().__init__(filename)
        self.init = init
        self.cond = cond
        self.update = update
        self.body = body
        self.lineno = lineno # Line of 'for'

class WhileStmtNode(Node):
    def __init__(self, cond, body, lineno, filename):
        super().__init__(filename)
        self.cond = cond
        self.body = body
        self.lineno = lineno # Line of 'while'
        
class DoWhileStmtNode(Node):
    def __init__(self, body, cond, lineno, filename):
        super().__init__(filename)
        self.body = body
        self.cond = cond
        self.lineno = lineno # Line of 'do'

class JumpStmtNode(Node):
    def __init__(self, jump_type, expr, lineno, filename):
        super().__init__(filename)
        self.jump_type = jump_type
        self.expr = expr
        self.lineno = lineno 

class ExprStmtNode(Node):
    def __init__(self, expr, lineno, filename):
        super().__init__(filename)
        self.expr = expr
        self.lineno = lineno # Line of ';'

class BinaryOpNode(Node):
    def __init__(self, left, op, right, lineno, filename):
        super().__init__(filename)
        self.left = left
        self.op = op
        self.right = right
        self.lineno = lineno # Line of operator

class UnaryOpNode(Node):
    def __init__(self, op, expr, lineno, filename):
        super().__init__(filename)
        self.op = op
        self.expr = expr
        self.lineno = lineno 
class AssignNode(Node):
    def __init__(self, lvalue, op, expr, lineno, filename):
        super().__init__(filename)
        self.lvalue = lvalue
        self.op = op          
        self.expr = expr
        self.lineno = lineno
        
class LiteralNode(Node):
    def __init__(self, value, literal_type, lineno, filename):
        super().__init__(filename)
        self.value = value
        self.literal_type = literal_type
        self.lineno = lineno

class FuncCallNode(Node):
    def __init__(self, name_node, args, lineno, filename):
        super().__init__(filename)
        self.name_node = name_node
        self.args = args
        self.lineno = lineno # Line of '('

class PostfixOpNode(Node):
    def __init__(self, expr, op, lineno, filename):
        super().__init__(filename)
        self.expr = expr
        self.op = op
        self.lineno = lineno # Line of '++' or '--'

class ArrayIndexNode(Node):
    def __init__(self, array, index, lineno, filename):
        super().__init__(filename)
        self.array = array
        self.index = index
        self.lineno = lineno # Line of '['

class StructMemberNode(Node):
    def __init__(self, struct_expr, member, lineno, filename):
        super().__init__(filename)
        self.struct_expr = struct_expr
        self.member = member
        self.lineno = lineno # Line of '.'

class TernaryOpNode(Node):
    def __init__(self, cond, true_expr, false_expr, lineno, filename):
        super().__init__(filename)
        self.cond = cond
        self.true_expr = true_expr
        self.false_expr = false_expr
        self.lineno = lineno # Line of '?'

class CastNode(Node):
    def __init__(self, target_type, expr, lineno, filename):
        super().__init__(filename)
        self.target_type = target_type
        self.expr = expr
        self.lineno = lineno # Line of '('

# --- Parser Implementation ---

# Get the token map from the lexer.
tokens = clexer.tokens
# ... (Rest of parser setup) ...
# Global variables to hold state during parsing
_parser_error_found = False
_parser_filename = ''


precedence = (
    ('right', 'ASSIGN', 'ADD_ASSIGN', 'SUB_ASSIGN', 'MUL_ASSIGN', 'DIV_ASSIGN', 'XOR_ASSIGN'),
    ('right', 'QUESTION_MARK', 'COLON'),
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'BITWISE_OR'),
    ('left', 'XOR'),
    ('left', 'BITWISE_AND'),
    ('left', 'EQUAL', 'NEQUAL'),
    ('left', 'LESS', 'LEQUAL', 'GREATER', 'GEQUAL'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'MODULO'),
    ('right', 'NOT', 'INCREMENT', 'DECREMENT', 'UMINUS', 'UPLUS'),
    ('left', 'DOT', 'LBRACKET', 'RPAREN'), # Note: LPAREN removed, handled by 'lvalue LPAREN'
    ('right', 'ELSE')
)


def p_program(p):
    'program : declaration_list'
    p[0] = ProgramNode(p[1], _parser_filename)

def p_declaration_list(p):
    '''declaration_list : declaration_list declaration
                        | empty'''
# ... (Grammar rules p_declaration_list to p_expression_cast) ...
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = []

def p_declaration(p):
    '''declaration : var_declaration
                   | func_definition
                   | func_prototype
                   | struct_definition'''
    p[0] = p[1]

def p_func_prototype(p):
    'func_prototype : type_specifier IDENTIFIER LPAREN params RPAREN SEMICOLON'
    name_node = IdentifierNode(p[2], p.lineno(2), _parser_filename)
    p[0] = FuncPrototypeNode(p[1], name_node, p[4], p.lineno(2), _parser_filename)

def p_struct_definition(p):
    'struct_definition : STRUCT IDENTIFIER LBRACE member_declaration_list RBRACE SEMICOLON'
    name_node = IdentifierNode(p[2], p.lineno(2), _parser_filename)
    p[0] = StructDefNode(name_node, p[4], p.lineno(2), _parser_filename)

def p_member_declaration_list(p):
    '''member_declaration_list : member_declaration_list member_declaration
                               | empty'''
    if len(p) == 3:
        p[0] = p[1] + p[2]
    else:
        p[0] = []

def p_member_declaration(p):
    'member_declaration : type_specifier declarator_list SEMICOLON'
    members = []
    for decl in p[2]:
        if isinstance(decl, InitDeclaratorNode):
            # C struct members cannot be initialized
            global _parser_error_found
            _parser_error_found = True
            sys.stderr.write(f"Parser error in file {_parser_filename} line {decl.lineno}: Initializer not allowed for struct member '{decl.identifier.name}'\n")
        else:
            members.append(MemberNode(p[1], decl, decl.lineno, _parser_filename))
    p[0] = members

def p_var_declaration(p):
    'var_declaration : type_specifier declarator_list SEMICOLON'
    p[0] = VarDeclNode(p[1], p[2], p.lineno(3), _parser_filename)

def p_declarator_list(p):
    '''declarator_list : declarator_list COMMA declarator
                       | declarator'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_declarator(p):
    '''declarator : IDENTIFIER
                  | IDENTIFIER LBRACKET INTEGER RBRACKET
                  | IDENTIFIER ASSIGN expression'''
    if len(p) == 2:
        # p[0] = IDENTIFIER
        p[0] = IdentifierNode(p[1], p.lineno(1), _parser_filename, is_array=False)
    elif p[2] == '=':
        # p[0] = IDENTIFIER ASSIGN expression
        ident = IdentifierNode(p[1], p.lineno(1), _parser_filename, is_array=False)
        p[0] = InitDeclaratorNode(ident, p[3], p.lineno(2), _parser_filename)
    else: 
        # p[0] = IDENTIFIER LBRACKET INTEGER RBRACKET (Array)
        p[0] = IdentifierNode(p[1], p.lineno(1), _parser_filename, is_array=True)


def p_func_definition(p):
    'func_definition : type_specifier IDENTIFIER LPAREN params RPAREN compound_statement'
    name_node = IdentifierNode(p[2], p.lineno(2), _parser_filename)
    p[0] = FuncDefNode(p[1], name_node, p[4], p[6], p.lineno(2), _parser_filename)

def p_params(p):
    '''params : param_list
              | empty'''
    p[0] = p[1] if p[1] is not None else []

def p_param_list(p):
    '''param_list : param_list COMMA param
                  | param'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_param_list_error(p):
    'param_list : param_list COMMA RPAREN'
    global _parser_error_found
    _parser_error_found = True
    
    sys.stderr.write(f"Parser error in file {_parser_filename} line {p.lineno(3)} at text {p[3]}\n")
    sys.stderr.write(f"  Description: Expected type name before ')'\n")
   
def p_param(p):
    '''param : type_specifier IDENTIFIER
             | type_specifier IDENTIFIER LBRACKET RBRACKET'''
    ident_node = IdentifierNode(p[2], p.lineno(2), _parser_filename)
    param_type = p[1]
    if len(p) == 5: # Array
        param_type = f"{p[1]}[]"
    p[0] = ParamNode(param_type, ident_node, p.lineno(1), _parser_filename)


def p_compound_statement(p):
    'compound_statement : LBRACE local_declarations statement_list RBRACE'
    p[0] = StmtBlockNode(p[2], p[3], p.lineno(1), _parser_filename)

def p_compound_statement(p):
    'compound_statement : LBRACE block_item_list RBRACE'
    
    declarations = []
    statements = []
    
    # We must sort the mixed list back into two separate lists
    # for the StmtBlockNode, which expects (var_declarations, statements).
    for item in p[2]:
        if item is None:
            continue # Skip empty expression_statements (';')

        # Check if the item is one of the nodes created by local_declaration
        if isinstance(item, (VarDeclNode, StructDefNode)):
            declarations.append(item)
        else:
            statements.append(item)
            
    p[0] = StmtBlockNode(declarations, statements, p.lineno(1), _parser_filename)

def p_block_item_list(p):
    '''block_item_list : block_item_list block_item
                       | empty'''
    if len(p) == 3:
        if p[2] is not None:
             p[0] = p[1] + [p[2]]
        else:
             p[0] = p[1]
    else:
        p[0] = []

def p_block_item(p):
    '''block_item : statement
                  | local_declaration'''
    # p_local_declaration is defined on line 311
    # p_statement is defined on line 327
    p[0] = p[1]

def p_local_declaration(p):
    '''local_declaration : var_declaration
                         | struct_definition'''
    p[0] = p[1]



def p_statement(p):
    '''statement : expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 | jump_statement'''
    p[0] = p[1]

def p_expression_statement(p):
    '''expression_statement : SEMICOLON
                            | expression SEMICOLON'''
    if len(p) == 2:
        p[0] = None # Empty statement
    else:
        p[0] = ExprStmtNode(p[1], p.lineno(2), _parser_filename)

def p_selection_statement_else(p):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement'
    p[0] = IfStmtNode(p[3], p[5], p[7], p.lineno(1), _parser_filename)

def p_selection_statement_no_else(p):
    'selection_statement : IF LPAREN expression RPAREN statement %prec ELSE'
    p[0] = IfStmtNode(p[3], p[5], None, p.lineno(1), _parser_filename)

def p_iteration_statement(p):
    '''iteration_statement : WHILE LPAREN expression RPAREN statement
                           | DO statement WHILE LPAREN expression RPAREN SEMICOLON
                           | FOR LPAREN expression_statement expression_statement expression_opt RPAREN statement'''
    if p[1] == 'while':
        p[0] = WhileStmtNode(p[3], p[5], p.lineno(1), _parser_filename)
    elif p[1] == 'do':
        p[0] = DoWhileStmtNode(p[2], p[5], p.lineno(1), _parser_filename)
    elif p[1] == 'for':
        p[0] = ForStmtNode(p[3], p[4], p[5], p[7], p.lineno(1), _parser_filename)


def p_jump_statement(p):
    '''jump_statement : CONTINUE SEMICOLON
                      | BREAK SEMICOLON
                      | RETURN SEMICOLON
                      | RETURN expression SEMICOLON'''
    if len(p) == 3:
        p[0] = JumpStmtNode(p[1], None, p.lineno(1), _parser_filename)
    else:
        p[0] = JumpStmtNode(p[1], p[2], p.lineno(1), _parser_filename)

def p_expression_assign(p):
    '''expression : lvalue ASSIGN expression
                  | lvalue ADD_ASSIGN expression
                  | lvalue SUB_ASSIGN expression
                  | lvalue MUL_ASSIGN expression
                  | lvalue DIV_ASSIGN expression
                  | lvalue XOR_ASSIGN expression'''
    # Pass p[2] (the operator) to the AssignNode constructor
    p[0] = AssignNode(p[1], p[2], p[3], p.lineno(2), _parser_filename)

def p_expression_postfix(p):
    '''expression : lvalue INCREMENT
                  | lvalue DECREMENT'''
    p[0] = PostfixOpNode(p[1], p[2], p.lineno(2), _parser_filename)

def p_lvalue(p):
    '''lvalue : IDENTIFIER
              | lvalue LBRACKET expression RBRACKET
              | lvalue DOT IDENTIFIER'''
    if len(p) == 2:
        p[0] = IdentifierNode(p[1], p.lineno(1), _parser_filename)
    elif p[2] == '[':
        p[0] = ArrayIndexNode(p[1], p[3], p.lineno(2), _parser_filename)
    elif p[2] == '.':
        member_node = IdentifierNode(p[3], p.lineno(3), _parser_filename)
        p[0] = StructMemberNode(p[1], member_node, p.lineno(2), _parser_filename)

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression STAR expression
                  | expression SLASH expression
                  | expression MODULO expression
                  | expression LESS expression
                  | expression LEQUAL expression
                  | expression GREATER expression
                  | expression GEQUAL expression
                  | expression EQUAL expression
                  | expression NEQUAL expression
                  | expression AND expression
                  | expression OR expression
                  | expression BITWISE_AND expression
                  | expression BITWISE_OR expression
                  | expression XOR expression'''
    p[0] = BinaryOpNode(p[1], p[2], p[3], p.lineno(2), _parser_filename)

def p_expression_unary(p):
    '''expression : NOT expression
                  | MINUS expression %prec UMINUS
                  | PLUS expression %prec UPLUS 
                  | INCREMENT lvalue
                  | DECREMENT lvalue'''
    p[0] = UnaryOpNode(p[1], p[2], p.lineno(1), _parser_filename)

def p_expression_ternary(p):
    'expression : expression QUESTION_MARK expression COLON expression'
    p[0] = TernaryOpNode(p[1], p[3], p[5], p.lineno(2), _parser_filename)

def p_expression_cast(p):
    'expression : LPAREN type_specifier RPAREN expression'
    p[0] = CastNode(p[2], p[4], p.lineno(1), _parser_filename)

def p_expression_call(p):
    'expression : lvalue LPAREN argument_list_opt RPAREN'
    # --- THIS IS THE FIX ---
    # Changed p[s[1]] to p[1]
    p[0] = FuncCallNode(p[1], p[3], p.lineno(2), _parser_filename)
    # --- END FIX ---

def p_argument_list_opt(p):
    '''argument_list_opt : argument_list
                         | empty'''
# ... (Grammar rules p_argument_list_opt to p_error) ...
    p[0] = p[1] if p[1] else []

def p_argument_list(p):
    '''argument_list : expression
                     | argument_list COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_expression_group(p):
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]

def p_expression_literal(p):
    '''expression : INTEGER
                  | REAL
                  | STRING
                  | CHARACTER
                  | TRUE
                  | FALSE'''
    # Note: 'true' and 'false' from lexer are type 'TRUE' and 'FALSE'
    p[0] = LiteralNode(p[1], p.slice[1].type, p.lineno(1), _parser_filename)
    
def p_expression_lvalue(p):
    'expression : lvalue'
    p[0] = p[1]

def p_expression_opt(p):
    '''expression_opt : expression
                      | empty'''
    p[0] = p[1]

def p_type_specifier(p):
    '''type_specifier : TYPE
                      | CONST TYPE
                      | TYPE CONST
                      | STRUCT IDENTIFIER
                      | CONST STRUCT IDENTIFIER
                      | STRUCT IDENTIFIER CONST
                      | BOOL
                      | CONST BOOL
                      | BOOL CONST
                      '''
    # This logic handles all combinations and standardizes
    # 'const' to be at the front.
    if 'const' in p:
        if p[1] == 'const':
            if len(p) == 3: # CONST TYPE or CONST BOOL
                p[0] = f"const {p[2]}"
            else: # CONST STRUCT IDENTIFIER
                p[0] = f"const {p[2]} {p[3]}"
        else: # TYPE CONST or STRUCT IDENTIFIER CONST or BOOL CONST
            if len(p) == 3: 
                p[0] = f"const {p[1]}"
            else:
                p[0] = f"const {p[1]} {p[2]}"
    else:
        if len(p) == 2: # TYPE or BOOL
            p[0] = p[1]
        else: # STRUCT IDENTIFIER
            p[0] = f"{p[1]} {p[2]}"

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    global _parser_error_found
    if _parser_error_found:
        return # Avoid cascading errors

    _parser_error_found = True
    if p:
        sys.stderr.write(f"Parser error in file {_parser_filename} line {p.lineno} at text {p.value}\n")
        sys.stderr.write(f"  Description: Unexpected token '{p.type}'\n")
    else:
        sys.stderr.write(f"Parser error in file {_parser_filename}: Unexpected end of file\n")

# --- Main Parser Functions ---

def build_ast(filename):
# ... (build_ast function) ...
    """
    Parses the source file and returns the AST.
    Returns None if parsing fails.
    """
    global _parser_error_found, _parser_filename
    _parser_error_found = False
    _parser_filename = filename

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file not found at '{filename}'\n")
        return None

    lexer = lex.lex(module=clexer)
    lexer.filename = filename
    
    # Ensure the 'parser' directory exists for parsetab.py
    parser_dir = os.path.dirname(__file__)
    if not parser_dir:
        parser_dir = '.'
        
    # --- THIS IS THE FIX ---
    # Changed tabmodule to force a *new* parser cache file
    parser = yacc.yacc(outputdir=parser_dir, tabmodule='my_parsetab_v4')
    # --- END FIX ---
    
    ast = parser.parse(source_code, lexer=lexer)

    if _parser_error_found:
        return None
    
    return ast

def write_parser_output(ast, filename):
# ... (write_parser_output function) ...
    """
    Takes a valid AST and writes the Phase 3 parser output file.
    """
    output_filename = os.path.splitext(filename)[0] + ".parser"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Traverse the AST to generate the required output
            for decl in ast.declarations:
                if isinstance(decl, VarDeclNode):
                    for declarator in decl.identifiers:
                        ident_node = declarator.identifier if isinstance(declarator, InitDeclaratorNode) else declarator
                        f.write(f"File {filename} Line {ident_node.lineno}: global variable {ident_node.name}\n")
                
                elif isinstance(decl, FuncDefNode):
                    f.write(f"File {filename} Line {decl.name_node.lineno}: function {decl.name_node.name}\n")
                    for param in decl.params:
                        f.write(f"File {filename} Line {param.ident_node.lineno}: parameter {param.ident_node.name}\n")
                    if decl.body and isinstance(decl.body, StmtBlockNode):
                       for local_decl in decl.body.var_declarations:
                           if isinstance(local_decl, VarDeclNode):
                               for declarator in local_decl.identifiers:
                                   ident_node = declarator.identifier if isinstance(declarator, InitDeclaratorNode) else declarator
                                   f.write(f"File {filename} Line {ident_node.lineno}: local variable {ident_node.name}\n")
                           elif isinstance(local_decl, StructDefNode):
                               f.write(f"File {filename} Line {local_decl.name_node.lineno}: local struct {local_decl.name_node.name}\n")
                               for member in local_decl.members:
                                   f.write(f"File {filename} Line {member.ident_node.lineno}: member {member.ident_node.name}\n")
                
                elif isinstance(decl, FuncPrototypeNode):
                    f.write(f"File {filename} Line {decl.name_node.lineno}: function {decl.name_node.name}\n")
                    for param in decl.params:
                        f.write(f"File {filename} Line {param.ident_node.lineno}: parameter {param.ident_node.name}\n")
                
                elif isinstance(decl, StructDefNode):
                    f.write(f"File {filename} Line {decl.name_node.lineno}: global struct {decl.name_node.name}\n")
                    for member in decl.members:
                        f.write(f"File {filename} Line {member.ident_node.lineno}: member {member.ident_node.name}\n")

        print(f"Successfully generated parser output at: {output_filename}")
    except IOError:
        sys.stderr.write(f"Error: Could not write to output file '{output_filename}'.\n")
        sys.exit(1)

def run_parser(filename):
# ... (run_parser function) ...
    ast = build_ast(filename)
    if ast:
        write_parser_output(ast, filename)
    else:
        print("Parsing failed, output file not generated.", file=sys.stderr)