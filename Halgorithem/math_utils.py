import ast
import math
import operator


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def eval_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        if type(node.op) not in OPS:
            raise ValueError("Unsupported math operator")
        return OPS[type(node.op)](eval_expr(node.left), eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_expr(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    raise ValueError("Unsafe expression")


def safe_eval(expr):
    tree = ast.parse(expr, mode="eval")
    return eval_expr(tree.body)


def numbers_close(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
