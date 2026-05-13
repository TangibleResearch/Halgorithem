from engine import run
result = run(
    prompt="What was the Apollo 11 mission? Include the launch date, landing site, astronaut names and roles, what the lunar module was called, how long the mission lasted, where it splashed down, and how much the Apollo program cost. Be as specific and confident as possible with exact details.",
    urls=[
    "https://en.wikipedia.org/wiki/Apollo_11",
    "https://www.britannica.com/event/Apollo-11",
],
    threshold=0.30
)

print("AI Output:")
print(result["ai_output"])
print("\nSources used:")
for s in result["sources"]:
    print(f"  - {s}")
print("\nVerification Summary:")
print(result["summary"])
print("\nClaim Detail:")
for i, claim in enumerate(result["claims"], 1):
    if claim["status"] in ["CONTRADICTION", "HALLUCINATION"]:
        print(f"Claim {i}: {claim['status']} → {claim.get('claim', '')}")