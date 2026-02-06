import sys
import os
from parser import (
    ProgramNode, VarDeclNode, FuncDefNode, StmtBlockNode, JumpStmtNode, ExprStmtNode,
    BinaryOpNode, UnaryOpNode, AssignNode, LiteralNode, FuncCallNode, IdentifierNode, 
    InitDeclaratorNode, ArrayIndexNode, CastNode, IfStmtNode, WhileStmtNode, 
    DoWhileStmtNode, ForStmtNode,TernaryOpNode
)

# --- JVM Type Mapping ---
JVM_TYPE_MAP = {
    'int': 'I', 'float': 'F', 'void': 'V', 'char': 'I', 'bool': 'I',
    'char[]': '[C', 'string': 'Ljava/lang/String;'
}

class CodeGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.classname = os.path.splitext(os.path.basename(filename))[0]
        self.output_lines = []
        
        self.global_vars = {} 
        self.global_arrays = {} 
        self.local_vars = {}  
        self.local_var_index = 0
        self.current_func_type = None
        self.func_signatures = {}
        
        # Phase 6: Label and Loop Management
        self.label_counter = 0
        self.loop_stack = [] # Stores tuples: (start_label, end_label)

    # --- Formatting Helpers ---
    
    def emit(self, line="", indent=2):
        prefix = "    " * indent if line and not line.endswith(':') else ""
        if line.endswith(':'): indent = 1 # Labels get less indent
        self.output_lines.append(f"{prefix}{line}")

    def emit_comment(self, msg, indent=2):
        self.emit(f"; {msg}", indent=indent)

    def new_label(self):
        self.label_counter += 1
        return f"L{self.label_counter}"

    def get_jvm_type(self, c_type):
        if c_type.endswith('[]') and c_type != 'char[]':
            return "[" + self.get_jvm_type(c_type[:-2])
        if c_type.startswith("const "):
            c_type = c_type.replace("const ", "").strip()
        return JVM_TYPE_MAP.get(c_type, 'I')

    def fmt_invoke(self, opcode, classname, method, signature):
        return f"{opcode} Method {classname} {method} {signature}"

    def fmt_field(self, opcode, classname, field, signature):
        return f"{opcode} Field {classname} {field} {signature}"

    def emit_push_int(self, val):
        val = int(val)
        if val == -1: self.emit("iconst_m1")
        elif 0 <= val <= 5: self.emit(f"iconst_{val}")
        elif -128 <= val <= 127: self.emit(f"bipush {val}")
        elif -32768 <= val <= 32767: self.emit(f"sipush {val}")
        else: self.emit(f"ldc {val}")

    def emit_push_float(self, val):
        fval = float(val)
        if fval == 0.0: self.emit("fconst_0")
        elif fval == 1.0: self.emit("fconst_1")
        elif fval == 2.0: self.emit("fconst_2")
        else: self.emit(f"ldc {val}f")

    def emit_load(self, v_type, idx, comment=""):
        op = 'fload' if v_type == 'float' else 'iload'
        suffix = f" ; {comment}" if comment else ""
        if 0 <= idx <= 3: self.emit(f"{op}_{idx}{suffix}")
        else: self.emit(f"{op} {idx}{suffix}")

    def emit_store(self, v_type, idx, comment=""):
        op = 'fstore' if v_type == 'float' else 'istore'
        suffix = f" ; {comment}" if comment else ""
        if 0 <= idx <= 3: self.emit(f"{op}_{idx}{suffix}")
        else: self.emit(f"{op} {idx}{suffix}")

    def estimate_stack(self, node):
        """
        Recursively calculates the maximum stack depth required by an AST node.
        Includes heuristics for control flow and boolean logic.
        """
        if node is None: return 0
        if isinstance(node, (LiteralNode, IdentifierNode)): return 1
        if isinstance(node, BinaryOpNode):
            # Short circuiting ops or comparisons might involve jumps/pushes
            return max(self.estimate_stack(node.left), 1 + self.estimate_stack(node.right)) + 1
        if isinstance(node, UnaryOpNode):
            return self.estimate_stack(node.expr) + 1
        if isinstance(node, AssignNode):
            rhs = self.estimate_stack(node.expr)
            if isinstance(node.lvalue, ArrayIndexNode):
                return max(self.estimate_stack(node.lvalue.array), self.estimate_stack(node.lvalue.index) + 1, rhs + 2) + 2
            return rhs + 1
        if isinstance(node, FuncCallNode):
            max_depth = 0
            for i, arg in enumerate(node.args):
                max_depth = max(max_depth, i + self.estimate_stack(arg))
            return max_depth + 1
        if isinstance(node, StmtBlockNode):
            m = 0
            for s in node.statements: m = max(m, self.estimate_stack(s))
            return m
        if isinstance(node, IfStmtNode):
            return max(self.estimate_stack(node.condition), self.estimate_stack(node.then_stmt), self.estimate_stack(node.else_stmt))
        if isinstance(node, (WhileStmtNode, DoWhileStmtNode)):
            return max(self.estimate_stack(node.cond), self.estimate_stack(node.body))
        if isinstance(node, ForStmtNode):
            return max(self.estimate_stack(node.init), self.estimate_stack(node.cond), self.estimate_stack(node.update), self.estimate_stack(node.body))
        if isinstance(node, TernaryOpNode):
            return max(self.estimate_stack(node.cond), self.estimate_stack(node.true_expr), self.estimate_stack(node.false_expr))
        return 1

    # --- Pass 1: Signature Collection ---
    def collect_signatures(self, ast):
        self.func_signatures['getint'] = '()I'
        self.func_signatures['getchar'] = '()I'
        self.func_signatures['putint'] = '(I)V'
        self.func_signatures['putchar'] = '(I)I'
        self.func_signatures['getfloat'] = '()F'
        self.func_signatures['putfloat'] = '(F)V'
        self.func_signatures['putstring'] = '([C)V'
        
        for decl in ast.declarations:
            if isinstance(decl, FuncDefNode):
                name = decl.name_node.name
                params = ""
                for p in decl.params:
                    params += self.get_jvm_type(p.param_type)
                ret = self.get_jvm_type(decl.return_type)
                self.func_signatures[name] = f"({params}){ret}"

    # --- Main Generation ---
    def generate(self, ast):
        self.collect_signatures(ast)
        self.emit(f".class public {self.classname}", indent=0)
        self.emit(".super java/lang/Object", indent=0)
        self.emit("", indent=0)

        for decl in ast.declarations:
            if isinstance(decl, VarDeclNode):
                self.visit_global_decl(decl)

        for decl in ast.declarations:
            if isinstance(decl, FuncDefNode):
                self.visit_FuncDefNode(decl)
        
        self.emit_clinit()
        self.emit_constructor()
        self.emit_java_main_wrapper()
        return "\n".join(self.output_lines)

    def emit_clinit(self):
        if not self.global_arrays: return
        self.emit(".method <clinit> : ()V", indent=0)
        self.emit(".code stack 1 locals 0", indent=1)
        for name, size in self.global_arrays.items():
            self.emit_comment(f"initializing variable {name}") 
            self.emit_push_int(size)
            self.emit("newarray int") 
            jvm_type = self.get_jvm_type(self.global_vars[name])
            self.emit(self.fmt_field("putstatic", self.classname, name, jvm_type))
        self.emit("return")
        self.emit(".end code", indent=1)
        self.emit(".end method", indent=0)
        self.emit("", indent=0)

    def emit_constructor(self):
        self.emit(".method <init> : ()V", indent=0)
        self.emit(".code stack 1 locals 1", indent=1)
        self.emit("aload_0")
        self.emit(self.fmt_invoke("invokespecial", "java/lang/Object", "<init>", "()V"))
        self.emit("return")
        self.emit(".end code", indent=1)
        self.emit(".end method", indent=0)

    def emit_java_main_wrapper(self):
        self.emit(".method public static main : ([Ljava/lang/String;)V", indent=0)
        self.emit(".code stack 1 locals 1", indent=1)
        self.emit(self.fmt_invoke("invokestatic", self.classname, "main", "()I"))
        self.emit(self.fmt_invoke("invokestatic", "java/lang/System", "exit", "(I)V"))
        self.emit("return")
        self.emit(".end code", indent=1)
        self.emit(".end method", indent=0)



    def visit_global_decl(self, node):
        base_type = node.var_type
        for decl in node.identifiers:
            if isinstance(decl, IdentifierNode):
                name, is_array = decl.name, decl.is_array
                if is_array:
                     size = getattr(decl, 'array_size', 10) 
                     self.global_arrays[name] = size
            elif isinstance(decl, InitDeclaratorNode):
                name, is_array = decl.identifier.name, decl.identifier.is_array
            
            c_type = f"{base_type}[]" if is_array else base_type
            jvm_type = self.get_jvm_type(c_type)
            self.global_vars[name] = c_type
            self.emit(f".field public static {name} {jvm_type}", indent=0)

    def visit_FuncDefNode(self, node):
        func_name = node.name_node.name
        self.current_func_type = node.return_type
        self.local_vars = {}
        self.local_var_index = 0
        
        params_sig = ""
        for param in node.params:
            jvm_type = self.get_jvm_type(param.param_type)
            params_sig += jvm_type
            self.local_vars[param.ident_node.name] = (self.local_var_index, param.param_type)
            self.local_var_index += 1

        ret_sig = self.get_jvm_type(node.return_type)
        self.emit(f".method public static {func_name} : ({params_sig}){ret_sig}", indent=0)
        
        self.prescan_locals(node.body)
        
        # Estimate stack + padding for complex logic
        stack_depth = self.estimate_stack(node.body) + 5
        
        self.emit(f".code stack {stack_depth} locals {self.local_var_index}", indent=1) 

        self.visit(node.body)

        # Handle implicit returns
        has_explicit_return = False
        if isinstance(node.body, StmtBlockNode) and node.body.statements:
            last = node.body.statements[-1]
            if isinstance(last, JumpStmtNode) and last.jump_type == 'return':
                has_explicit_return = True
        
        end_line = node.body.statements[-1].lineno + 1 if node.body.statements else node.lineno + 1
        self.emit_comment(f"implicit return at {self.filename} line {end_line}")

        if node.return_type == 'void':
            if not has_explicit_return: self.emit("return")
            else: self.emit(";DEAD return") 
        else:
            if not has_explicit_return:
                # Main is a special case in some C compilers, but strict C requires return.
                if func_name == 'main':
                    self.emit("iconst_0")
                    self.emit("ireturn")
                else:
                    self.emit("iconst_0") # Default return to satisfy verifier
                    self.emit("ireturn")
            else:
                self.emit(";DEAD return")

        self.emit(".end code", indent=1)
        self.emit(".end method", indent=0)
        self.emit("", indent=0)

    def prescan_locals(self, node):
        if not node: return
        if isinstance(node, VarDeclNode):
            base_type = node.var_type
            for item in node.identifiers:
                name = item.name if isinstance(item, IdentifierNode) else item.identifier.name
                if name not in self.local_vars:
                    self.local_vars[name] = (self.local_var_index, base_type)
                    self.local_var_index += 1
        if isinstance(node, StmtBlockNode):
            for decl in node.var_declarations: self.prescan_locals(decl)
            for stmt in node.statements: self.prescan_locals(stmt)
        
        if isinstance(node, IfStmtNode):
            self.prescan_locals(node.then_stmt)
            self.prescan_locals(node.else_stmt)
        if isinstance(node, (WhileStmtNode, DoWhileStmtNode)):
            self.prescan_locals(node.body)
        if isinstance(node, ForStmtNode):
            self.prescan_locals(node.init)
            self.prescan_locals(node.body)

    # --- Dispatcher ---
    def visit(self, node):
        if node is None: return
        method_name = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        sys.stderr.write(f"Warning: No visitor for {node.__class__.__name__}\n")

    # --- Statement Visitors ---

    def visit_StmtBlockNode(self, node):
        for decl in node.var_declarations:
            if isinstance(decl, VarDeclNode):
                base_type = decl.var_type
                for item in decl.identifiers:
                    if isinstance(item, InitDeclaratorNode):
                        name = item.identifier.name
                        self.visit(item.expression)
                        idx, _ = self.local_vars[name]
                        self.emit_store(base_type, idx, name) 
        for stmt in node.statements:
            self.visit(stmt)

    def visit_IfStmtNode(self, node):
        line = node.lineno
        self.emit_comment(f"if statement at {self.filename} line {line}")
        
        else_label = self.new_label()
        end_label = self.new_label()

        # Evaluate condition
        self.visit(node.condition)
        self.emit(f"ifeq {else_label}")

        # Then block
        self.visit(node.then_stmt)
        self.emit(f"goto {end_label}")

        # Else block
        self.emit(f"{else_label}:")
        if node.else_stmt:
            self.visit(node.else_stmt)
        
        self.emit(f"{end_label}:")

    def visit_WhileStmtNode(self, node):
        line = node.lineno
        self.emit_comment(f"while loop at {self.filename} line {line}")
        
        start_label = self.new_label()
        end_label = self.new_label()
        
        # Push loop context for break/continue
        self.loop_stack.append((start_label, end_label))

        self.emit(f"{start_label}:")
        
        # Condition
        self.visit(node.cond)
        self.emit(f"ifeq {end_label}")
        
        # Body
        self.visit(node.body)
        self.emit(f"goto {start_label}")
        
        self.emit(f"{end_label}:")
        self.loop_stack.pop()

    def visit_DoWhileStmtNode(self, node):
        line = node.lineno
        self.emit_comment(f"do-while loop at {self.filename} line {line}")
        
        start_label = self.new_label()
        cond_label = self.new_label() # For continue to jump to
        end_label = self.new_label()
        
        self.loop_stack.append((cond_label, end_label))

        self.emit(f"{start_label}:")
        
        # Body
        self.visit(node.body)
        
        # Label for continue to hit before condition check
        self.emit(f"{cond_label}:")
        
        # Condition
        self.visit(node.cond)
        self.emit(f"ifne {start_label}") # If true, jump back to start
        
        self.emit(f"{end_label}:")
        self.loop_stack.pop()

    def visit_ForStmtNode(self, node):
        line = node.lineno
        self.emit_comment(f"for loop at {self.filename} line {line}")
        
        start_label = self.new_label()
        update_label = self.new_label()
        end_label = self.new_label()
        
        # Init
        if node.init:
            self.visit(node.init)
        
        self.loop_stack.append((update_label, end_label))

        self.emit(f"{start_label}:")
        
        # Condition
        if node.cond:
             if isinstance(node.cond, ExprStmtNode) and node.cond.expr is None:
                 pass 
             else:
                 
                 # Visit the expression directly so the value stays on the stack.
                 if isinstance(node.cond, ExprStmtNode):
                     self.visit(node.cond.expr)
                 else:
                     self.visit(node.cond)
                     
                 self.emit(f"ifeq {end_label}")

        # Body
        self.visit(node.body)
        
        # Update
        self.emit(f"{update_label}:")
        if node.update:
            expr_type = self.visit(node.update)
            
            if expr_type != 'void' and not isinstance(node.update, ExprStmtNode):
                 self.emit("pop")
        
        self.emit(f"goto {start_label}")
        
        self.emit(f"{end_label}:")
        self.loop_stack.pop()

    def visit_JumpStmtNode(self, node):
        line = node.lineno
        self.emit_comment(f"{node.jump_type} at {self.filename} line {line}")
        
        if node.jump_type == 'return':
            if node.expr:
                self.visit(node.expr)
                ret_op = 'freturn' if self.current_func_type == 'float' else 'ireturn'
                self.emit(ret_op)
            else:
                self.emit("return")
        
        elif node.jump_type == 'break':
            if not self.loop_stack:
                print(f"Code generation error in file {self.filename} line {line}", file=sys.stderr)
                print("Description: break not inside a loop", file=sys.stderr)
                
            else:
                _, end_label = self.loop_stack[-1]
                self.emit(f"goto {end_label}")

        elif node.jump_type == 'continue':
            if not self.loop_stack:
                print(f"Code generation error in file {self.filename} line {line}", file=sys.stderr)
                print("Description: continue not inside a loop", file=sys.stderr)
            else:
                start_label, _ = self.loop_stack[-1]
                self.emit(f"goto {start_label}")

    def visit_ExprStmtNode(self, node):
        if node.expr:
            line = node.expr.lineno if hasattr(node.expr, 'lineno') else 0
            self.emit_comment(f"expression statement at {self.filename} line {line}")
            if isinstance(node.expr, AssignNode):
                self.visit_AssignNode(node.expr, is_statement=True)
            else:
                expr_type = self.visit(node.expr)
                if expr_type != 'void': self.emit("pop")

    def visit_AssignNode(self, node, is_statement=False):
        if isinstance(node.lvalue, ArrayIndexNode):
            self.visit(node.lvalue.array)
            self.visit(node.lvalue.index)
            if node.op != '=': 
                self.emit("dup2")
                self.emit("iaload")
                self.visit(node.expr)
                if node.op == '+=': self.emit("iadd")
                elif node.op == '-=': self.emit("isub")
                elif node.op == '*=': self.emit("imul")
                elif node.op == '/=': self.emit("idiv")
                elif node.op == '^=': self.emit("ixor")
            else:
                self.visit(node.expr) 

            if is_statement:
                self.emit("iastore")
                return 'void'
            else:
                self.emit("dup_x2")
                self.emit("iastore")
                return 'int'
        else:
            name = node.lvalue.name
            target_type = 'int'
            idx = -1
            is_local = False

            if name in self.local_vars:
                idx, target_type = self.local_vars[name]
                is_local = True
            elif name in self.global_vars:
                target_type = self.global_vars[name]
                is_local = False
            
            if node.op != '=':
                if is_local:
                    self.emit_load(target_type, idx, name)
                else:
                    jvm_type = self.get_jvm_type(target_type)
                    self.emit(self.fmt_field("getstatic", self.classname, name, jvm_type))
            
            self.visit(node.expr)

            if node.op != '=':
                op_prefix = 'f' if target_type == 'float' else 'i'
                if node.op == '+=': self.emit(f"{op_prefix}add")
                elif node.op == '-=': self.emit(f"{op_prefix}sub")
                elif node.op == '*=': self.emit(f"{op_prefix}mul")
                elif node.op == '/=': self.emit(f"{op_prefix}div")
                elif node.op == '^=': self.emit("ixor")

            if not is_statement: self.emit("dup")
            
            if is_local:
                self.emit_store(target_type, idx, name)
            else:
                jvm_type = self.get_jvm_type(target_type)
                self.emit(self.fmt_field("putstatic", self.classname, name, jvm_type))
            
            return target_type

    # --- Expression Visitors ---

    def visit_CastNode(self, node):
        expr_type = self.visit(node.expr)
        target = node.target_type
        if expr_type == 'int' and target == 'float':
            self.emit("i2f")
            return 'float'
        elif expr_type == 'float' and target == 'int':
            self.emit("f2i")
            return 'int'
        return target

    def visit_ArrayIndexNode(self, node):
        self.visit(node.array)
        self.visit(node.index)
        self.emit("iaload")
        return 'int' 

    def visit_LiteralNode(self, node):
        if node.literal_type == 'INTEGER':
            self.emit_push_int(node.value)
            return 'int'
        elif node.literal_type == 'REAL':
            self.emit_push_float(node.value)
            return 'float'
        elif node.literal_type == 'STRING':
            self.emit(f"ldc {node.value}")
            self.emit(self.fmt_invoke("invokestatic", "lib440", "java2c", "(Ljava/lang/String;)[C"))
            return 'char[]'
        elif node.literal_type == 'CHARACTER':
            char_val = node.value
            val = ord(char_val[1]) if len(char_val) >= 3 else 0
            self.emit_push_int(val)
            return 'int'
        elif node.literal_type == 'TRUE':
            self.emit("iconst_1")
            return 'int'
        elif node.literal_type == 'FALSE':
            self.emit("iconst_0")
            return 'int'

    def visit_IdentifierNode(self, node):
        name = node.name
        if name in self.local_vars:
            idx, v_type = self.local_vars[name]
            self.emit_load(v_type, idx, name)
            return v_type
        elif name in self.global_vars:
            v_type = self.global_vars[name]
            jvm_type = self.get_jvm_type(v_type)
            self.emit(self.fmt_field("getstatic", self.classname, name, jvm_type))
            return v_type

    def visit_UnaryOpNode(self, node):
        op = node.op
        
        # Handle Prefix Increment/Decrement (++i, --i)
        if op == '++' or op == '--':
            if isinstance(node.expr, IdentifierNode):
                name = node.expr.name
                
                # Case 1: Local Variable (Use optimized iinc)
                if name in self.local_vars:
                    idx, type_str = self.local_vars[name]
                    amount = 1 if op == '++' else -1
                    
                    # 1. Update the variable in the local table
                    self.emit(f"iinc {idx} {amount}")
                    # 2. Load the NEW value onto the stack (Prefix behavior)
                    self.emit_load(type_str, idx, name)
                    return type_str
                
                # Case 2: Global Variable
                elif name in self.global_vars:
                    type_str = self.global_vars[name]
                    jvm_type = self.get_jvm_type(type_str)
                    
                    # 1. Load current value
                    self.emit(self.fmt_field("getstatic", self.classname, name, jvm_type))
                    # 2. Add/Sub 1
                    self.emit("iconst_1")
                    if op == '++': self.emit("iadd")
                    else: self.emit("isub")
                    # 3. Store new value back
                    self.emit("dup") 
                    self.emit(self.fmt_field("putstatic", self.classname, name, jvm_type))
                    return type_str
                    
            return 'int' 

        
        expr_type = self.visit(node.expr)
        
        if op == '-':
            if expr_type == 'float': self.emit("fneg")
            else: self.emit("ineg")
            return expr_type
            
        elif op == '!':
            # Logical NOT: if val == 0 -> 1, else -> 0
            true_lbl = self.new_label()
            end_lbl = self.new_label()
            self.emit(f"ifeq {true_lbl}") # if 0, jump to true
            self.emit("iconst_0")          # was not 0, so push 0
            self.emit(f"goto {end_lbl}")
            self.emit(f"{true_lbl}:")
            self.emit("iconst_1")
            self.emit(f"{end_lbl}:")
            return 'int'
            
        return expr_type
    def visit_BinaryOpNode(self, node):
        op = node.op
        
        # --- Short Circuit Logic ---
        if op == '&&':
            false_lbl = self.new_label()
            end_lbl = self.new_label()
            
            self.visit(node.left)
            self.emit(f"ifeq {false_lbl}") # If left is 0 (false), jump to false
            
            self.visit(node.right)
            self.emit(f"ifeq {false_lbl}") # If right is 0, jump to false
            
            self.emit("iconst_1") # Both true
            self.emit(f"goto {end_lbl}")
            
            self.emit(f"{false_lbl}:")
            self.emit("iconst_0")
            
            self.emit(f"{end_lbl}:")
            return 'int'
            
        elif op == '||':
            true_lbl = self.new_label()
            end_lbl = self.new_label()
            
            self.visit(node.left)
            self.emit(f"ifne {true_lbl}") # If left is 1 (true), jump to true
            
            self.visit(node.right)
            self.emit(f"ifne {true_lbl}") # If right is 1, jump to true
            
            self.emit("iconst_0") # Both false
            self.emit(f"goto {end_lbl}")
            
            self.emit(f"{true_lbl}:")
            self.emit("iconst_1")
            
            self.emit(f"{end_lbl}:")
            return 'int'

        # --- Standard Arithmetic/Relational ---
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        is_float = (left_type == 'float' or right_type == 'float')
        
        if op in ['+', '-', '*', '/', '%', '&', '|', '^']:
            op_prefix = 'f' if is_float else 'i'
            if op == '+': self.emit(f"{op_prefix}add")
            elif op == '-': self.emit(f"{op_prefix}sub")
            elif op == '*': self.emit(f"{op_prefix}mul")
            elif op == '/': self.emit(f"{op_prefix}div")
            elif op == '%': self.emit("frem" if is_float else "irem")
            elif op == '&': self.emit("iand")
            elif op == '|': self.emit("ior")
            elif op == '^': self.emit("ixor")
            return 'float' if is_float else 'int'
        
        # --- Comparisons ---
        # All comparisons result in 'int' (0 or 1)
        true_label = self.new_label()
        end_label = self.new_label()
        
        if is_float:
            self.emit("fcmpl") # Compare floats: -1, 0, 1
            
            if op == '==': self.emit(f"ifeq {true_label}")
            elif op == '!=': self.emit(f"ifne {true_label}")
            elif op == '<': self.emit(f"iflt {true_label}") 
            elif op == '<=': self.emit(f"ifle {true_label}")
            elif op == '>': self.emit(f"ifgt {true_label}")
            elif op == '>=': self.emit(f"ifge {true_label}")
        else:
            
            if op == '==': self.emit(f"if_icmpeq {true_label}")
            elif op == '!=': self.emit(f"if_icmpne {true_label}")
            elif op == '<': self.emit(f"if_icmplt {true_label}")
            elif op == '<=': self.emit(f"if_icmple {true_label}")
            elif op == '>': self.emit(f"if_icmpgt {true_label}")
            elif op == '>=': self.emit(f"if_icmpge {true_label}")
            
        self.emit("iconst_0")
        self.emit(f"goto {end_label}")
        self.emit(f"{true_label}:")
        self.emit("iconst_1")
        self.emit(f"{end_label}:")
        
        return 'int'
    def visit_TernaryOpNode(self, node):
        false_label = self.new_label()
        end_label = self.new_label()

        
        self.visit(node.cond)
        self.emit(f"ifeq {false_label}") 

        # 2. True Path
        true_type = self.visit(node.true_expr)
        self.emit(f"goto {end_label}")

        # 3. False Path
        self.emit(f"{false_label}:")
        false_type = self.visit(node.false_expr)

        self.emit(f"{end_label}:")
        
        
        return true_type
    def visit_FuncCallNode(self, node):
        name = node.name_node.name
        for arg in node.args: self.visit(arg)
        
        
        std_lib_funcs = {
            'getint', 'getchar', 'putint', 'putchar', 
            'getfloat', 'putfloat', 'putstring'
        }
        
        if name in self.func_signatures:
            sig = self.func_signatures[name]
            
            
            if name in std_lib_funcs:
                cls = "lib440"
            else:
                cls = self.classname
                
            self.emit(self.fmt_invoke("invokestatic", cls, name, sig))
            
            ret_char = sig.split(')')[-1]
            if ret_char == 'V': return 'void'
            if ret_char == 'F': return 'float'
            return 'int'
        else:
            
            self.emit(self.fmt_invoke("invokestatic", self.classname, name, "()I"))
            return 'int'

def run_code_generator(filename, ast):
    generator = CodeGenerator(filename)
    try:
        jvm_code = generator.generate(ast)
        output_filename = os.path.splitext(filename)[0] + ".j"
        with open(output_filename, 'w', newline='\n') as f:
            f.write(jvm_code)
        print(f"Successfully generated assembly at: {output_filename}")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")