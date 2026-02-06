import sys
import os
import parser as cparser 
from parser import (
    Node, ProgramNode, VarDeclNode, InitDeclaratorNode, FuncDefNode,
    FuncPrototypeNode, ParamNode, IdentifierNode, StructDefNode, MemberNode,
    StmtBlockNode, IfStmtNode, ForStmtNode, WhileStmtNode, DoWhileStmtNode,
    JumpStmtNode, ExprStmtNode, BinaryOpNode, UnaryOpNode, AssignNode,
    LiteralNode, FuncCallNode, PostfixOpNode, ArrayIndexNode, StructMemberNode,
    TernaryOpNode, CastNode
)

# --- Type System Definitions ---

T_INT = "int"
T_FLOAT = "float"
T_CHAR = "char"
T_VOID = "void"
T_BOOL = "bool"
T_ERROR = "error_type"
T_STRING = "char[]" 

SCALAR_TYPES = {T_INT, T_FLOAT, T_CHAR, T_BOOL}
NUMERIC_TYPES = {T_INT, T_FLOAT}
INTEGRAL_TYPES = {T_INT, T_CHAR, T_BOOL}

# --- Type Compatibility Rules ---

BINARY_OP_RULES = {
    # Arithmetic
    ('int', '+', 'int'): 'int',
    ('float', '+', 'float'): 'float',
    ('int', '+', 'float'): 'float',
    ('float', '+', 'int'): 'float',
    
    ('int', '-', 'int'): 'int',
    ('float', '-', 'float'): 'float',
    ('int', '-', 'float'): 'float',
    ('float', '-', 'int'): 'float',
    
    ('int', '*', 'int'): 'int',
    ('float', '*', 'float'): 'float',
    ('int', '*', 'float'): 'float',
    ('float', '*', 'int'): 'float',
    
    ('int', '/', 'int'): 'int',
    ('float', '/', 'float'): 'float',
    ('int', '/', 'float'): 'float',
    ('float', '/', 'int'): 'float',
    
    ('int', '%', 'int'): 'int',
    
    # Relational
    ('int', '<', 'int'): 'char',
    ('float', '<', 'float'): 'char',
    ('int', '<', 'float'): 'char',
    ('float', '<', 'int'): 'char',
    ('char', '<', 'char'): 'char',
    
    ('int', '<=', 'int'): 'char',
    ('float', '<=', 'float'): 'char',
    ('int', '<=', 'float'): 'char',
    ('float', '<=', 'int'): 'char',
    ('char', '<=', 'char'): 'char',
    
    ('int', '>', 'int'): 'char',
    ('float', '>', 'float'): 'char',
    ('int', '>', 'float'): 'char',
    ('float', '>', 'int'): 'char',
    ('char', '>', 'char'): 'char',
    
    ('int', '>=', 'int'): 'char',
    ('float', '>=', 'float'): 'char',
    ('int', '>=', 'float'): 'char',
    ('float', '>=', 'int'): 'char',
    ('char', '>=', 'char'): 'char',

    # Equality
    ('int', '==', 'int'): 'char',
    ('float', '==', 'float'): 'char',
    ('int', '==', 'float'): 'char',
    ('float', '==', 'int'): 'char',
    ('bool', '==', 'bool'): 'char',
    ('char', '==', 'char'): 'char',
    
    ('int', '!=', 'int'): 'char',
    ('float', '!=', 'float'): 'char',
    ('int', '!=', 'float'): 'char',
    ('float', '!=', 'int'): 'char',
    ('bool', '!=', 'bool'): 'char',
    ('char', '!=', 'char'): 'char',
    
    # Logical
    ('bool', '&&', 'bool'): 'char',
    ('bool', '||', 'bool'): 'char',
    
    # Bitwise
    ('int', '&', 'int'): 'int',
    ('int', '|', 'int'): 'int',
    ('int', '^', 'int'): 'int',
    ('bool', '^', 'bool'): 'bool',
}

# --- Symbol Table ---

class Symbol:
    def __init__(self, name, type, kind, lineno, is_const=False, param_types=None, is_defined=False):
        self.name = name
        self.type = type       
        self.kind = kind       
        self.lineno = lineno
        self.is_const = is_const
        self.return_type = type 
        self.param_types = param_types if param_types is not None else []
        self.is_defined = is_defined

class Scope:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def add_symbol(self, symbol):
        if symbol.name in self.symbols:
            return False
        self.symbols[symbol.name] = symbol
        return True

    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def lookup_current(self, name):
        return self.symbols.get(name)

# --- AST Visitor ---

class TypeChecker:
    def __init__(self, filename):
        self.filename = filename
        self.global_scope = Scope()
        
        # --- NEW: Initialize Standard Library ---
        self.initialize_std_library()
        
        self.current_scope = self.global_scope
        self.struct_table = {}
        self.current_function = None
        self.errors = []
        self.type_log = [] 

    def initialize_std_library(self):
        """Hard-codes the lib440 functions into the global scope."""
        # int getchar()
        self.global_scope.add_symbol(Symbol("getchar", "int", "function", 0, param_types=[], is_defined=True))
        # int putchar(int c)
        self.global_scope.add_symbol(Symbol("putchar", "int", "function", 0, param_types=["int"], is_defined=True))
        # int getint()
        self.global_scope.add_symbol(Symbol("getint", "int", "function", 0, param_types=[], is_defined=True))
        # void putint(int x)
        self.global_scope.add_symbol(Symbol("putint", "void", "function", 0, param_types=["int"], is_defined=True))
        # float getfloat()
        self.global_scope.add_symbol(Symbol("getfloat", "float", "function", 0, param_types=[], is_defined=True))
        # void putfloat(float x)
        self.global_scope.add_symbol(Symbol("putfloat", "void", "function", 0, param_types=["float"], is_defined=True))
        # void putstring(const char s[])
        # Note: We use "char[]" to ensure compatibility with string literals
        self.global_scope.add_symbol(Symbol("putstring", "void", "function", 0, param_types=["char[]"], is_defined=True))

    def log_type(self, lineno, kind, name, type_str):
        self.type_log.append(f"File {self.filename} Line {lineno}: {kind} {name} has type {type_str}")

    def add_error(self, lineno, message):
        self.errors.append(f"Type checking error in file {self.filename} line {lineno}: {message}")

    def enter_scope(self):
        self.current_scope = Scope(parent=self.current_scope)

    def exit_scope(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent
        else:
            self.add_error(0, "Compiler error: Exited global scope.")

    def is_valid_type(self, type_str):
        if type_str.startswith("const "):
            type_str = type_str.split(" ", 1)[1]
            
        if type_str in SCALAR_TYPES or type_str == T_VOID:
            return True
        if type_str.startswith("struct "):
            struct_name = type_str.split(" ", 1)[1]
            if struct_name in self.struct_table:
                return True
        if type_str.endswith("[]"): 
             base_type = type_str[:-2]
             return self.is_valid_type(base_type)
        return False

    def check_assignment(self, target_type, expr_type, lineno):
        if target_type == T_ERROR or expr_type == T_ERROR:
            return True 
            
        if target_type == expr_type:
            return True 
        
        # Strip const for comparison
        target_base = self.strip_const(target_type)
        expr_base = self.strip_const(expr_type)
        
        if target_base == expr_base:
            return True

        if target_base == T_FLOAT and expr_base in {T_INT, T_CHAR, T_BOOL}:
            return True
            
        if target_base == T_INT and expr_base in {T_FLOAT, T_CHAR, T_BOOL}:
            return True
            
        if target_base in SCALAR_TYPES and expr_base in SCALAR_TYPES:
             return True
             
        if target_type.startswith("struct ") or target_type.endswith("[]"):
            self.add_error(lineno, f"Invalid assignment: '{target_type}'='{expr_type}'")
            return False

        self.add_error(lineno, f"Type mismatch: Cannot assign type '{expr_type}' to '{target_type}'")
        return False

    def check_condition(self, cond_type, lineno):
        if cond_type == T_ERROR:
            return
        if cond_type not in SCALAR_TYPES:
            node_lineno = lineno if isinstance(lineno, int) else 0 
            self.add_error(node_lineno, f"Type mismatch: Condition must be a scalar type (int, float, char, or bool), not '{cond_type}'")

    def strip_const(self, type_str):
        if isinstance(type_str, str) and type_str.startswith("const "):
            return type_str[6:]
        return type_str

    def can_widen(self, from_type, to_type):
        from_type = self.strip_const(from_type)
        to_type = self.strip_const(to_type)
        
        if from_type == to_type:
            return True
        if from_type == T_CHAR and to_type == T_INT:
            return True
        if from_type == T_BOOL and to_type == T_INT:
            return True
        if from_type == T_CHAR and to_type == T_FLOAT:
            return True
        if from_type == T_BOOL and to_type == T_FLOAT:
            return True
        if from_type == T_INT and to_type == T_FLOAT:
            return True
            
        return False

    def visit(self, node):
        if node is None:
            return None
        method_name = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        print(f"Warning: No visitor for node type {node.__class__.__name__}", file=sys.stderr)
        return T_ERROR

    def visit_ProgramNode(self, node):
        for decl in node.declarations:
            if isinstance(decl, StructDefNode):
                self.visit_StructDefNode(decl) 
            elif isinstance(decl, FuncDefNode):
                self.register_function(decl, is_prototype=False)
            elif isinstance(decl, FuncPrototypeNode):
                self.register_function(decl, is_prototype=True)
        
        for decl in node.declarations:
            if isinstance(decl, VarDeclNode):
                self.visit_VarDeclNode(decl, kind="global variable")
            elif isinstance(decl, FuncDefNode):
                self.visit_FuncDefNode(decl)
            elif isinstance(decl, (FuncPrototypeNode, StructDefNode)):
                pass 
        
        if not self.errors:
            return self.type_log
        else:
            for err in self.errors:
                sys.stderr.write(f"{err}\n")
            return None

    def visit_StructDefNode(self, node):
        struct_name = node.name_node.name
        lineno = node.name_node.lineno
        
        if struct_name in self.struct_table:
            self.add_error(lineno, f"Re-definition of 'struct {struct_name}'")
            return
            
        struct_members = {}
        for member in node.members:
            self.visit_MemberNode(member, struct_members)
            
        self.struct_table[struct_name] = struct_members

    def visit_MemberNode(self, node, struct_members):
        base_type = node.member_type
        member_name = node.ident_node.name
        lineno = node.ident_node.lineno
        
        if not self.is_valid_type(base_type):
            self.add_error(lineno, f"Unknown type '{base_type}' for member '{member_name}'")
            base_type = T_ERROR
            
        if member_name in struct_members:
            self.add_error(lineno, f"Duplicate member '{member_name}' in struct")

        is_array = node.ident_node.is_array
        final_type = f"{base_type}[]" if is_array else base_type
        
        struct_members[member_name] = final_type

    def register_function(self, node, is_prototype):
        func_name = node.name_node.name
        lineno = node.name_node.lineno
        return_type = node.return_type
        
        is_definition = not is_prototype
        
        if not self.is_valid_type(return_type):
            self.add_error(lineno, f"Unknown return type '{return_type}' for function '{func_name}'")
            return_type = T_ERROR

        param_types = []
        param_names = set()
        for param in node.params:
            param_type = param.param_type
            param_name = param.ident_node.name
            
            if not self.is_valid_type(param_type):
                self.add_error(param.ident_node.lineno, f"Unknown type '{param_type}' for parameter '{param_name}'")
                param_type = T_ERROR
            
            if param_name in param_names:
                self.add_error(param.ident_node.lineno, f"Duplicate parameter name '{param_name}' in function '{func_name}'")
            
            param_types.append(param_type)
            param_names.add(param_name)

        existing_sym = self.current_scope.lookup_current(func_name)
        if existing_sym:
            if existing_sym.kind != "function":
                self.add_error(lineno, f"'{func_name}' is already defined as a {existing_sym.kind}")
                return

            if existing_sym.return_type != return_type or existing_sym.param_types != param_types:
                self.add_error(lineno, f"Conflicting types for function '{func_name}'")
                return 

            if is_definition and existing_sym.is_defined:
                self.add_error(lineno, f"Duplicate definition for function '{func_name}'")
                return
            
            if is_definition:
                existing_sym.is_defined = True
            
        else:
            func_sym = Symbol(func_name, return_type, "function", lineno, param_types=param_types, is_defined=is_definition)
            self.current_scope.add_symbol(func_sym)

    def visit_FuncDefNode(self, node):
        func_name = node.name_node.name
        self.current_function = self.current_scope.lookup(func_name)
        
        if not self.current_function:
            return

        self.enter_scope()
        
        for param in node.params:
            self.visit(param)
            
        self.visit(node.body)
        
        self.exit_scope()
        self.current_function = None

    def visit_ParamNode(self, node):
        name = node.ident_node.name
        type_str = node.param_type
        lineno = node.ident_node.lineno
        is_const = type_str.startswith("const ")
        
        sym = Symbol(name, type_str, "parameter", lineno, is_const=is_const)
        if not self.current_scope.add_symbol(sym):
            self.add_error(lineno, f"Duplicate parameter name '{name}'")

    def visit_VarDeclNode(self, node, kind="local variable"):
        kind = "local variable" if self.current_scope.parent else "global variable"
        
        base_type = node.var_type
        
        base_type_stripped = self.strip_const(base_type)
        if base_type_stripped == T_VOID:
            for declarator in node.identifiers:
                if isinstance(declarator, InitDeclaratorNode):
                    lineno = declarator.identifier.lineno
                    name = declarator.identifier.name
                else:
                    lineno = declarator.lineno
                    name = declarator.name
                self.add_error(lineno, f"Variable '{name}' cannot be of type 'void'")
            return 

        if not self.is_valid_type(base_type):
            first_ident_lineno = node.identifiers[0].lineno if node.identifiers else 0
            self.add_error(first_ident_lineno, f"Unknown type '{base_type}'")
            base_type = T_VOID
 
        for decl in node.identifiers:
            is_array = False
            if isinstance(decl, IdentifierNode):
                is_array = decl.is_array
            elif isinstance(decl, InitDeclaratorNode):
                is_array = decl.identifier.is_array

            final_type = f"{base_type}[]" if is_array else base_type
            is_const = final_type.startswith("const ")
            
            if isinstance(decl, IdentifierNode):
                name = decl.name
                lineno = decl.lineno
                
                existing_sym = self.current_scope.lookup_current(name)
                if existing_sym:
                    self.add_error(lineno, f"'{name}' is already defined in this scope (as a {existing_sym.kind})")
                    continue 

                sym = Symbol(name, final_type, kind, lineno, is_const=is_const)
                self.current_scope.add_symbol(sym) 

            elif isinstance(decl, InitDeclaratorNode):
                name = decl.identifier.name
                lineno = decl.identifier.lineno
                
                existing_sym = self.current_scope.lookup_current(name)
                if existing_sym:
                    self.add_error(lineno, f"'{name}' is already defined in this scope (as a {existing_sym.kind})")
                
                expr_type = self.visit(decl.expression)
                
                if is_const and expr_type is None: 
                         self.add_error(lineno, f"Constant variable '{name}' must be initialized with a valid expression")
                         
                self.check_assignment(final_type, expr_type, lineno)
                
                sym = Symbol(name, final_type, kind, lineno, is_const=is_const)
                
                if not existing_sym:
                    self.current_scope.add_symbol(sym)

    def visit_StmtBlockNode(self, node):
        self.enter_scope()
        for decl in node.var_declarations:
            if isinstance(decl, VarDeclNode):
                self.visit_VarDeclNode(decl, kind="local variable")
            elif isinstance(decl, StructDefNode):
                self.visit_StructDefNode(decl)
        
        for stmt in node.statements:
            self.visit(stmt)
        self.exit_scope()

    def visit_IfStmtNode(self, node):
        cond_type = self.visit(node.condition)
        self.check_condition(cond_type, node.condition.lineno if hasattr(node.condition, 'lineno') else node.filename)
        
        self.visit(node.then_stmt)
        if node.else_stmt:
            self.visit(node.else_stmt)

    def visit_WhileStmtNode(self, node):
        cond_type = self.visit(node.cond)
        self.check_condition(cond_type, node.cond.lineno if hasattr(node.cond, 'lineno') else node.filename)
        self.visit(node.body)

    def visit_DoWhileStmtNode(self, node):
        self.visit(node.body)
        cond_type = self.visit(node.cond)
        self.check_condition(cond_type, node.cond.lineno if hasattr(node.cond, 'lineno') else node.filename)

    def visit_ForStmtNode(self, node):
        self.enter_scope()
        if node.init:
            self.visit(node.init)
            
        if node.cond:
            # Check for empty condition (infinite loop)
            if isinstance(node.cond, ExprStmtNode) and node.cond.expr is None:
                pass 
            else:
                # FIX: Unwrap ExprStmtNode. 
                # We need the type of the expression inside the statement.
                if isinstance(node.cond, ExprStmtNode):
                    cond_type = self.visit(node.cond.expr)
                else:
                    cond_type = self.visit(node.cond)

                self.check_condition(cond_type, node.cond.lineno if hasattr(node.cond, 'lineno') else node.filename)
                
        if node.update:
            self.visit(node.update)
            
        self.visit(node.body)
        self.exit_scope()

    def visit_JumpStmtNode(self, node):
        node_lineno = node.lineno

        if node.jump_type == 'return':
            if not self.current_function:
                self.add_error(node_lineno, "Return statement outside of a function")
                return
            
            target_type = self.current_function.return_type
            
            if node.expr:
                expr_type = self.visit(node.expr)
                if target_type == T_VOID:
                    self.add_error(node_lineno, f"Function '{self.current_function.name}' is 'void' and should not return a value")
                else:
                    self.check_assignment(target_type, expr_type, node_lineno)
            else: 
                if target_type != T_VOID:
                    self.add_error(node_lineno, f"Function '{self.current_function.name}' must return a value of type '{target_type}'")
    
    def visit_ExprStmtNode(self, node):
        if node.expr:
            expr_type = self.visit(node.expr)
            if expr_type != T_ERROR:
                line = node.expr.lineno
                self.type_log.append(f"File {self.filename} Line {line}: expression has type {expr_type}")

    def visit_AssignNode(self, node):
        lvalue_sym = None
        if isinstance(node.lvalue, IdentifierNode):
            lvalue_sym = self.current_scope.lookup(node.lvalue.name)
            
        if lvalue_sym and lvalue_sym.is_const:
             self.add_error(node.lvalue.lineno, f"Cannot assign to constant variable '{lvalue_sym.name}'")

        lvalue_type = self.visit(node.lvalue)
        expr_type = self.visit(node.expr)
        
        self.check_assignment(lvalue_type, expr_type, node.lvalue.lineno)
        
        return lvalue_type

    def visit_BinaryOpNode(self, node):
        original_left_type = self.visit(node.left)
        original_right_type = self.visit(node.right)
        op = node.op
        
        if original_left_type == T_ERROR or original_right_type == T_ERROR:
            return T_ERROR
            
        left_type = self.strip_const(original_left_type)
        right_type = self.strip_const(original_right_type)

        key = None
        
        if op == '&&' or op == '||':
            if left_type in SCALAR_TYPES: left_type = T_BOOL
            if right_type in SCALAR_TYPES: right_type = T_BOOL
            key = (left_type, op, right_type)
        
        elif left_type == T_CHAR and right_type == T_CHAR:
            if op in {'+', '-', '*', '/', '%', '&', '|', '^'}:
                return T_CHAR 
            elif op in {'<', '<=', '>', '>=', '==', '!='}:
                key = (left_type, op, right_type) 
            else:
                key = (left_type, op, right_type) 
        else:
            if left_type in INTEGRAL_TYPES: left_type = T_INT
            if right_type in INTEGRAL_TYPES: right_type = T_INT
            
            if (left_type in NUMERIC_TYPES) and (right_type in NUMERIC_TYPES):
                if left_type == T_FLOAT or right_type == T_FLOAT:
                     key = (T_FLOAT, op, T_FLOAT)
                else:
                     key = (T_INT, op, T_INT)
            else:
                key = (left_type, op, right_type)
        
        result_type = BINARY_OP_RULES.get(key)
        
        if result_type is None:
            self.add_error(node.left.lineno, f"Invalid operands: Cannot apply operator '{op}' to types '{original_left_type}' and '{original_right_type}'")
            return T_ERROR
            
        return result_type

    def visit_UnaryOpNode(self, node):
        op = node.op
        expr_type = self.visit(node.expr)
        
        if expr_type == T_ERROR:
            return T_ERROR
            
        if op == '!':
            if expr_type not in SCALAR_TYPES:
                self.add_error(node.expr.lineno, f"Invalid operand: Cannot apply logical NOT '!' to non-scalar type '{expr_type}'")
                return T_ERROR
            return T_CHAR
            
        if op == '-' or op == '+':
            if expr_type in INTEGRAL_TYPES:
                return T_INT
            if expr_type == T_FLOAT:
                return T_FLOAT
                
            self.add_error(node.expr.lineno, f"Invalid operand: Cannot apply unary '{op}' to non-numeric type '{expr_type}'")
            return T_ERROR
            
        if op == '++' or op == '--':
            if isinstance(node.expr, IdentifierNode):
                sym = self.current_scope.lookup(node.expr.name)
                if sym and sym.is_const:
                    self.add_error(node.expr.lineno, f"Cannot modify constant variable '{sym.name}' with '{op}'")
            
            if expr_type not in INTEGRAL_TYPES and expr_type != T_FLOAT:
                self.add_error(node.expr.lineno, f"Invalid operand: Cannot apply '{op}' to non-numeric type '{expr_type}'")
                return T_ERROR
            return expr_type 

        return T_ERROR

    def visit_PostfixOpNode(self, node):
        op = node.op
        expr_type = self.visit(node.expr)

        if expr_type == T_ERROR:
            return T_ERROR
            
        if isinstance(node.expr, IdentifierNode):
            sym = self.current_scope.lookup(node.expr.name)
            if sym and sym.is_const:
                self.add_error(node.expr.lineno, f"Cannot modify constant variable '{sym.name}' with '{op}'")
        
        if expr_type not in INTEGRAL_TYPES and expr_type != T_FLOAT:
            self.add_error(node.expr.lineno, f"Invalid operand: Cannot apply '{op}' to non-numeric type '{expr_type}'")
            return T_ERROR
            
        return expr_type 

    def visit_TernaryOpNode(self, node):
        cond_type = self.visit(node.cond)
        self.check_condition(cond_type, node.cond.lineno if hasattr(node.cond, 'lineno') else node.filename)
        
        true_type = self.visit(node.true_expr)
        false_type = self.visit(node.false_expr)
        
        if true_type == T_ERROR or false_type == T_ERROR:
            return T_ERROR
            
        if self.check_assignment(true_type, false_type, node.true_expr.lineno):
             return true_type
        elif self.check_assignment(false_type, true_type, node.false_expr.lineno):
             return false_type
        else:
             self.add_error(node.cond.lineno, f"Type mismatch: Incompatible types in ternary operator: '{true_type}' and '{false_type}'")
             return T_ERROR

    def visit_CastNode(self, node):
        target_type = node.target_type
        expr_type = self.visit(node.expr)
        
        if not self.is_valid_type(target_type):
            self.add_error(node.expr.lineno, f"Invalid type cast: Unknown target type '{target_type}'")
            return T_ERROR
            
        if expr_type == T_ERROR:
            return T_ERROR
            
        if target_type in SCALAR_TYPES and expr_type in SCALAR_TYPES:
            return target_type
        
        self.add_error(node.expr.lineno, f"Invalid type cast: Cannot cast from '{expr_type}' to '{target_type}'")
        return T_ERROR

    def visit_FuncCallNode(self, node):
        if not isinstance(node.name_node, IdentifierNode):
            self.add_error(node.name_node.lineno, "Function calls must use a simple identifier name")
            return T_ERROR
            
        func_name = node.name_node.name
        lineno = node.name_node.lineno
        
        func_sym = self.current_scope.lookup(func_name)
        
        if not func_sym:
            self.add_error(lineno, f"Call to undeclared function '{func_name}'")
            return T_ERROR
        
        if func_sym.kind != "function":
            self.add_error(lineno, f"'{func_name}' is not a function, it is a {func_sym.kind}")
            return T_ERROR
            
        arg_types = [self.visit(arg) for arg in node.args]
        param_types = func_sym.param_types
        
        if len(arg_types) != len(param_types):
            self.add_error(lineno, f"Wrong number of arguments for function '{func_name}': expected {len(param_types)}, received {len(arg_types)}")
            return func_sym.return_type
            
        for i, (arg_type, param_type) in enumerate(zip(arg_types, param_types)):
            arg_lineno = node.args[i].lineno if hasattr(node.args[i], 'lineno') else lineno
            self.check_assignment(param_type, arg_type, arg_lineno)
            
        return func_sym.return_type

    def visit_ArrayIndexNode(self, node):
        array_type = self.visit(node.array)
        index_type = self.visit(node.index)
        
        if array_type == T_ERROR or index_type == T_ERROR:
            return T_ERROR
            
        if index_type != T_INT:
            self.add_error(node.index.lineno, f"Array index must be an integer, not '{index_type}'")
            
        if not array_type.endswith("[]"):
            self.add_error(node.array.lineno, f"Cannot apply array index to non-array type '{array_type}'")
            return T_ERROR
            
        return array_type[:-2]

    def visit_StructMemberNode(self, node):
     struct_expr_type = self.visit(node.struct_expr)
    
     if struct_expr_type == T_ERROR:
        return T_ERROR
        
     is_struct_const = struct_expr_type.startswith("const ")
    
     base_struct_type = struct_expr_type
     if is_struct_const:
        base_struct_type = struct_expr_type[6:] 
    
     if not base_struct_type.startswith("struct "):
        self.add_error(node.struct_expr.lineno, f"Cannot access member of non-struct type '{struct_expr_type}'")
        return T_ERROR
        
     struct_name = base_struct_type.split(" ", 1)[1]
     struct_def = self.struct_table.get(struct_name)
    
     if not struct_def:
        self.add_error(node.struct_expr.lineno, f"Using incomplete or unknown struct type '{struct_name}'")
        return T_ERROR
        
     member_name = node.member.name
     member_type = struct_def.get(member_name)
    
     if not member_type:
        self.add_error(node.member.lineno, f"Struct '{struct_name}' has no member named '{member_name}'")
        return T_ERROR
        
     if is_struct_const and not member_type.startswith("const "):
        return f"const {member_type}"
     else:
        return member_type
        
    def visit_IdentifierNode(self, node):
        name = node.name
        lineno = node.lineno
        
        sym = self.current_scope.lookup(name)
        
        if not sym:
            self.add_error(lineno, f"Undeclared identifier '{name}'")
            return T_ERROR
            
        if sym.kind == "function":
            self.add_error(lineno, f"Cannot use function '{name}' as a variable")
            return T_ERROR
            
        return sym.type

    def visit_LiteralNode(self, node):
        type_map = {
            'INTEGER': T_INT,
            'REAL': T_FLOAT,
            'STRING': T_STRING,
            'CHARACTER': T_CHAR,
            'BOOL': T_BOOL,
            'TRUE': T_BOOL,
            'FALSE': T_BOOL 
        }
        return type_map.get(node.literal_type, T_ERROR)


def run_type_checker(filename, ast):
    checker = TypeChecker(filename)
    type_log = checker.visit(ast)
    
    if type_log:
        output_filename = os.path.splitext(filename)[0] + ".types"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(type_log))
                f.write("\n")
            print(f"Successfully generated type file at: {output_filename}")
            return True
        except IOError:
            sys.stderr.write(f"Error: Could not write to output file '{output_filename}'.\n")
            return False
    else:
        print("Type checking failed due to errors.", file=sys.stderr)
        return False