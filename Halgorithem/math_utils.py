import re

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
MAX_EXPR_LENGTH = 120
ALLOWED_EXPR_RE = re.compile(r"^[\d\s+\-*/().%^eE]+$")


def safe_eval(expr):
    expr = str(expr).strip().replace("^", "**")
    if len(expr) > MAX_EXPR_LENGTH:
        raise ValueError("Expression too long")
    if not ALLOWED_EXPR_RE.fullmatch(expr):
        raise ValueError("Expression contains unsupported symbols")
    if "**" in expr:
        for exponent in re.findall(r"\*\*\s*(\d+)", expr):
            if int(exponent) > 12:
                raise ValueError("Exponent too large")
    try:
        result = parse_expr(
            expr,
            transformations=TRANSFORMATIONS,
            local_dict={},
            global_dict={
                "__builtins__": {},
                "Integer": sympy.Integer,
                "Float": sympy.Float,
                "Rational": sympy.Rational,
            },
            evaluate=True,
        )
        return float(result.evalf())
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {expr}") from e


def numbers_close(left, right, rel_tol=1e-6):
    return sympy.Abs(sympy.Float(left) - sympy.Float(right)) <= rel_tol * max(sympy.Abs(sympy.Float(left)), sympy.Abs(sympy.Float(right)), sympy.Float(1))
