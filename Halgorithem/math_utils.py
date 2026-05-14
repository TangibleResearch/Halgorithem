import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def safe_eval(expr):
    try:
        result = parse_expr(str(expr), transformations=TRANSFORMATIONS)
        return float(result.evalf())
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {expr}") from e


def numbers_close(left, right, rel_tol=1e-6):
    return sympy.Abs(sympy.Float(left) - sympy.Float(right)) <= rel_tol * max(sympy.Abs(sympy.Float(left)), sympy.Abs(sympy.Float(right)), sympy.Float(1))