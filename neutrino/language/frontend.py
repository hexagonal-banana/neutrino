"""Parse and flatten Python Tracing DSL"""
from neutrino.language import TYPES # neutrino/language/__init__.py
import ast
from typing import Optional
from dataclasses import dataclass
from neutrino.common import Register, Probe, Map

allowed_nodes = {
    ast.Import,     # Imported Stuff
    ast.Module,     # the greatest start
    ast.Name,       # Name of Variable
    ast.Assign,     # Assign Value
    ast.AugAssign,  # +=
    ast.UnaryOp,    # Unary Op, only negative
    ast.BinOp,      # Binary Op, +-*/
    ast.Call,       # Call function
    ast.Attribute,  # Access Attribute of Namespace
    ast.Constant,   # Constant Value
    ast.Expr,       # Single Expression
}

binary_ops = {
    ast.Add:  "add",
    ast.Sub:  "sub",
    ast.Mult: "mul",
    ast.Div:  "div"
}

unary_ops = {
    ast.USub: "neg"
}

class NeutrinoVisitor(ast.NodeVisitor):
    def __init__(self, nl_name: str, regs: list[str], maps: list[str]):
        super().__init__()
        self.nl_name = nl_name
        self.reg_counter = -1 # make it R0
        self.ir: list[tuple] = []
        self.reg_map: dict[str, str] = {reg: self.fresh_name() for reg in regs}
        self.maps = maps
        # initialize and visit tree

    def fresh_name(self):
        self.reg_counter += 1
        return f"NR{self.reg_counter}"

    def visit_Assign(self, node): # Lowered to mov 
        # we shall check if the target has a known name
        name = self.reg_map[node.targets[0].id]
        if isinstance(node.value, (ast.Attribute, ast.Constant)):
            self.ir.append(["mov", name, self.process_operand(node.value)])
        else:
            new_name = self.visit(node.value) # this is the temporary name
            for inst in self.ir:
                for idx in range(len(inst)):
                    if inst[idx] == new_name:
                        inst[idx] = name
            self.reg_counter -= 1

    def process_operand(self, operand) -> str:
        if isinstance(operand, ast.Name):
            return self.reg_map[operand.id]
        elif isinstance(operand, ast.Constant):
            return operand.value
        elif isinstance(operand, (ast.Attribute, ast.Call, ast.BinOp, ast.UnaryOp)):
            return self.visit(operand)
        else:
            raise ValueError
    
    def visit_BinOp(self, node): # Lowered to add/sub
        lhs = self.process_operand(node.left)
        rhs = self.process_operand(node.right)
        new_name = self.fresh_name()
        if isinstance(node.op, ast.Add):
            self.ir.append(["add", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Sub):
            self.ir.append(["sub", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Mult):
            self.ir.append(["mul", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Div):
            self.ir.append(["div", new_name, lhs, rhs])
        elif isinstance(node.op, ast.LShift):
            self.ir.append(["lsh", new_name, lhs, rhs])
        else:
            raise NotImplementedError()
        self.reg_map[new_name] = new_name
        return new_name
    
    def visit_AugAssign(self, node):
        rhs = self.process_operand(node.value)
        name = self.reg_map[node.target.id]
        if isinstance(node.op, ast.Add):
            self.ir.append(["add", name, name, rhs])
        elif isinstance(node.op, ast.Sub):
            self.ir.append(["sub", name, name, rhs])
        elif isinstance(node.op, ast.Mult):
            self.ir.append(["mul", name, name, rhs])
        elif isinstance(node.op, ast.Div):
            self.ir.append(["div", name, name, rhs])
        elif isinstance(node.op, ast.LShift):
            self.ir.append(["lsh", name, name, rhs])
        else:
            raise NotImplementedError()
        return name

    def visit_UnaryOp(self, node): 
        value = self.process_operand(node.operand)
        new_name = self.fresh_name()
        if isinstance(node.op, ast.USub):
            self.ir.append(["neg", new_name, value])
        else:
            raise NotImplementedError()
        return new_name

    def visit_Call(self, node):
        func_name = self.visit(node.func)
        if func_name == "cuid":
            new_name = self.fresh_name()
            self.ir.append(["cuid", new_name])
            self.reg_map[new_name] = new_name
            return new_name
        elif func_name == "time":
            new_name = self.fresh_name()
            self.reg_map[new_name] = new_name
            self.ir.append(["time", new_name]) 
            return new_name
        elif func_name == "clock":
            new_name = self.fresh_name()
            self.reg_map[new_name] = new_name
            self.ir.append(["clock", new_name]) 
            return new_name
        elif func_name == "save":
            map_name = node.func.value.id
            if map_name not in self.maps:
                raise ValueError(f"Map {map_name} not found, known maps: {self.maps}")            
            regs = [self.process_operand(arg) for arg in node.args]
            self.ir.append(["SAVE", map_name] + regs)
        else:
            raise NotImplementedError()
    
    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        if node.value.id == self.nl_name or node.value.id in self.maps:
            if node.attr == "site_id":
                return "SITE_ID"
            if node.attr == "event":
                return "EVENT_KIND"
            if node.attr in ("bytes", "addr", "out", "in1", "in2", "in3"):
                return node.attr.upper()
            return node.attr
        else:
            raise ValueError(f"can only refer to neutrino.language semantic but got {node.value.id}")

    def visit_Constant(self, node):
        return node

    def generic_visit(self, node):
        if type(node) not in allowed_nodes:
            raise NotImplementedError(f"{type(node).__name__} (lineno: {node.lineno})")
        super().generic_visit(node)


def parse(code: str, *, register_descriptions=None) -> tuple[int, list[Probe], list[Map], str]:
    """Parse the code into probes"""
    tree = ast.parse(code)
    nl_name:  str = None # name of neutrino.language in the code
    regs:     list[Register] = []
    num_regs: int = 0
    probes:   list[Probe] = []
    callback: str = "" # not used yet, but we can use it later
    maps:     list[Map] = [] # not used yet, but we can use it later

    for node in tree.body:
        if type(node) is ast.Import and node.names[0].name == "neutrino.language":
            nl_name = node.names[0].asname
        elif type(node) is ast.Assign and node.targets[0].id == "CALLBACK":
            if isinstance(node.value, ast.Constant):
                callback = node.value.value
            else:
                raise ValueError("CALLBACK must be a string constant")
        elif type(node) is ast.AnnAssign and node.annotation:
            if node.annotation.value.id == nl_name and node.annotation.attr in TYPES:
                initial = ast.literal_eval(node.value) if node.value is not None else 0
                if type(initial) is not int:
                    raise ValueError("Register initializers must be integers")
                regs.append(Register(node.target.id, node.annotation.attr, initial))
        elif type(node) is ast.ClassDef and node.decorator_list:
            name = node.name # take class name as map name
            decorator = node.decorator_list[0]
            if decorator.func.id == "Map":
                level, type_, size, cap, contents = None, None, 0, 1, []
                for keyword in decorator.keywords:
                    if   keyword.arg == "level": level = keyword.value.value
                    elif keyword.arg == "type":  type_ = keyword.value.value
                    elif keyword.arg == "size":  size = keyword.value.value
                    elif keyword.arg == "cap":   cap = keyword.value.value
                if size % 8 != 0: 
                    raise ValueError("size must be multiple of 8 to avoid misaligned address")
                if not level or not type_: 
                    raise ValueError("level and type must be specified")
                if not isinstance(cap, int) and cap != "dynamic": 
                    raise ValueError("cap must be an integer or 'dynamic'")
                # check if map existed or not
                for node in node.body:
                    if type(node) is ast.AnnAssign and node.annotation:
                        if node.annotation.value.id == nl_name and node.annotation.attr in TYPES:
                            contents.append(Register(node.target.id, node.annotation.attr, None))
                    else:
                        raise ValueError(f"Map {name} must only contain AnnAssign nodes")
                ordered = sorted(contents, key=lambda reg: reg.dtype, reverse=True)
                if ordered != contents:
                    print("[warn] map contents reordered")
                # create a map object 
                maps.append(Map(name=name, level=level, type=type_, size=size, cap=cap, regs=ordered))
        elif type(node) is ast.FunctionDef and node.decorator_list:
            name = node.name # take func name as probe name
            decorator = node.decorator_list[0]
            if decorator.func.id == "probe":
                pos, level, before, funcname = None, "thread", False, None
                if decorator.args:
                    if len(decorator.args) > 4:
                        raise ValueError("probe accepts at most four positional arguments")
                    positional = dict(zip(("pos", "after", "level", "size"),
                                          (ast.literal_eval(arg) for arg in decorator.args)))
                    pos = positional.get("pos")
                    level = positional.get("level", level)
                    if any(k.arg in positional for k in decorator.keywords):
                        raise ValueError("Duplicate positional and keyword probe arguments")
                for keyword in decorator.keywords:
                    if   keyword.arg == "pos":       pos = keyword.value.value
                    elif keyword.arg == "level":     level = keyword.value.value
                    elif keyword.arg == "before":    before = keyword.value.value
                    elif keyword.arg == "funcname":  funcname = keyword.value.value
                if not pos or not level: raise ValueError("position must be specified")
                # check if probe existed or not
                visitor = NeutrinoVisitor(nl_name=nl_name, regs=[reg.name for reg in regs], maps=[map.name for map in maps])
                visitor.visit(ast.Module(body=node.body)) # Take it as independent code
                probe = Probe(name=name, pos=pos, level=level, funcname=funcname)
                if before:
                    probe.before = visitor.ir
                else:
                    probe.after  = visitor.ir
                probes.append(probe)
                num_regs = max(num_regs, visitor.reg_counter)
    
    if register_descriptions is not None:
        register_descriptions.extend(regs)
    return num_regs + len(regs), probes, maps, callback

# A Simple Test Case, not really used in production
if __name__ == "__main__":
    import sys

    code = open(sys.argv[1], "r").read()

    regs, probes, maps, callback = parse(code)
    
    print(regs)
    print(probes)
    print(maps)
    print(callback)
