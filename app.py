from flask import Flask, render_template, request
from ai import generate
from engine import verify

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    ai_output = None
    results = None

    if request.method == "POST":
        query = request.form.get("query")

        ai_output = generate(query)
        results = verify(ai_output)
    print("AI OUTPUT:", ai_output)
    print("RESULTS:", results)
    return render_template(
        "index.html",
        ai_output=ai_output,
        results=results
    )

if __name__ == "__main__":
    app.run(debug=True)