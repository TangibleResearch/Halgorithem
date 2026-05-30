import re

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
MAX_EXPR_LENGTH = 120
MAX_EXPONENT_ABS = 12
ALLOWED_EXPR_RE = re.compile(r"^[\d\s+\-*/().%^eE]+$")


def _validate_expr(expr):
    expr = str(expr).strip()
    if not expr:
        raise ValueError("Empty expression")
    if len(expr) > MAX_EXPR_LENGTH:
        raise ValueError("Expression too long")
    if not ALLOWED_EXPR_RE.fullmatch(expr):
        raise ValueError("Expression contains unsupported characters")
    for exponent in re.findall(r"(?:\*\*|\^)\s*([+-]?\d+(?:\.\d+)?)", expr):
        if abs(float(exponent)) > MAX_EXPONENT_ABS:
            raise ValueError("Exponent too large")
    return expr.replace("^", "**")


def safe_eval(expr):
    try:
        expr = _validate_expr(expr)
        result = parse_expr(expr, transformations=TRANSFORMATIONS, evaluate=True)
        return float(result.evalf())
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {expr}") from e


def numbers_close(left, right, rel_tol=1e-6):
    return sympy.Abs(sympy.Float(left) - sympy.Float(right)) <= rel_tol * max(sympy.Abs(sympy.Float(left)), sympy.Abs(sympy.Float(right)), sympy.Float(1))
