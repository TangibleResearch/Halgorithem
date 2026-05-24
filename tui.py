import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()

def parse_threshold(raw_threshold):
    try:
        threshold = float(raw_threshold)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError
    except (TypeError, ValueError):
        return 0.30
    return threshold


def is_valid_threshold(raw_threshold):
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return False
    return 0.0 <= threshold <= 1.0


def parse_positive_int(raw_value, default, minimum=1):
    try:
        return max(minimum, int(raw_value))
    except (TypeError, ValueError):
        return default


def parse_csv(raw_value):
    return [item.strip() for item in (raw_value or "").split(",") if item.strip()]


def collect_inputs():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Halgorithem[/bold cyan] [dim]©2026 Tangible Research[/dim]\n"
        "[dim]AI Output Verification Engine[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

    existing_key = os.environ.get("OPENAI_API_KEY", "")
    if existing_key:
        console.print(f"[dim]Using existing OPENAI_API_KEY ({existing_key[:4]}...)[/dim]")
    else:
        api_key = Prompt.ask("[bold green]OpenAI API key[/bold green]", password=True)
        if not api_key.strip():
            console.print("[red]No API key provided. Exiting.[/red]")
            raise SystemExit(1)
        os.environ["OPENAI_API_KEY"] = api_key
        console.print(f"[dim green]Key set ({api_key[:4]}...)[/dim green]")

    console.print()
    console.print(Rule("[bold]Source Documents[/bold]", style="cyan"))
    console.print()

    source_mode = Prompt.ask(
        "[bold blue]Source type[/bold blue]",
        choices=["urls", "files", "both"],
        default="urls",
    )

    urls, truth_file_paths = [], []

    if source_mode in ("urls", "both"):
        urls = parse_csv(Prompt.ask("[bold blue]URLs[/bold blue] [dim](comma-separated)[/dim]", default=""))
        if urls:
            console.print(f"  [dim]→ {len(urls)} URL(s) queued[/dim]")

    if source_mode in ("files", "both"):
        truth_file_paths = parse_csv(Prompt.ask("[bold blue]File paths[/bold blue] [dim](comma-separated)[/dim]", default=""))
        if truth_file_paths:
            console.print(f"  [dim]→ {len(truth_file_paths)} file(s) queued[/dim]")

    if not urls and not truth_file_paths:
        console.print("[red]No sources provided. Exiting.[/red]")
        raise SystemExit(1)

    console.print()
    console.print(Rule("[bold]Settings[/bold]", style="cyan"))
    console.print()

    raw_threshold = Prompt.ask(
        "[bold magenta]Verification threshold[/bold magenta] [dim](0.0–1.0)[/dim]",
        default="0.30",
    )
    threshold = parse_threshold(raw_threshold)
    if not is_valid_threshold(raw_threshold):
        console.print("[yellow]Invalid threshold — defaulting to 0.30[/yellow]")

    raw_chunks = Prompt.ask(
        "[bold magenta]Sentences per chunk[/bold magenta]",
        default="2",
    )
    sentences_per_chunk = parse_positive_int(raw_chunks, default=2, minimum=1)

    raw_overlap = Prompt.ask(
        "[bold magenta]Sentence overlap[/bold magenta]",
        default="1",
    )
    sentence_overlap = parse_positive_int(raw_overlap, default=1, minimum=0)

    console.print()
    console.print(Rule("[bold]Query[/bold]", style="cyan"))
    console.print()
    prompt = Prompt.ask("[bold yellow]Your prompt[/bold yellow]")
    console.print()

    return {
        "prompt": prompt,
        "urls": urls,
        "truth_file_paths": truth_file_paths,
        "threshold": threshold,
        "sentences_per_chunk": sentences_per_chunk,
        "sentence_overlap": sentence_overlap,
    }


def run_engine(config, engine_class=None):
    if engine_class is None:
        from engine import Engine
        engine_class = Engine

    eng = engine_class(
        sentences_per_chunk=config["sentences_per_chunk"],
        sentence_overlap=config["sentence_overlap"],
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scraping sources...", total=None)
        source_docs = eng._load_sources(config["urls"], config["truth_file_paths"])

        progress.update(task, description="Generating AI response...")
        ai_output = eng.generate(config["prompt"], source_docs)

        progress.update(task, description="Verifying claims...")
        verification = eng.verify(ai_output, source_docs, threshold=config["threshold"])

    return source_docs, ai_output, verification


def render_results(source_docs, ai_output, verification, config):
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Results[/bold cyan]",
        border_style="cyan",
        padding=(0, 4),
    ))
    console.print()

# AI Output
    console.print(Rule("[bold]AI Output[/bold]", style="dim"))
    console.print()
    console.print(ai_output)
    console.print()

# Sources
    console.print(Rule("[bold]Sources[/bold]", style="dim"))
    console.print()
    for s in source_docs:
        console.print(f"  [dim cyan]•[/dim cyan] {s['file_path']}")
    console.print()

# Settings used
    console.print(Rule("[bold]Run Config[/bold]", style="dim"))
    console.print()
    config_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    config_table.add_column(style="dim")
    config_table.add_column(style="bold white")
    config_table.add_row("Threshold", str(config["threshold"]))
    config_table.add_row("Sentences/chunk", str(config["sentences_per_chunk"]))
    config_table.add_row("Overlap", str(config["sentence_overlap"]))
    config_table.add_row("Sources", str(len(source_docs)))
    console.print(config_table)
    console.print()

# Summary
    claims = verification["claims"]
    total = len(claims)
    supported = sum(1 for c in claims if c["status"] == "SUPPORTED")
    weak = sum(1 for c in claims if c["status"] == "WEAK_SUPPORT")
    contradicts = sum(1 for c in claims if c["status"] == "CONTRADICTION")
    hallucinated = sum(1 for c in claims if c["status"] == "HALLUCINATION")
    denials = sum(1 for c in claims if c["status"] == "UNVERIFIABLE_DENIAL")

    console.print(Rule("[bold]Verification Summary[/bold]", style="dim"))
    console.print()
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary_table.add_column(style="dim")
    summary_table.add_column(justify="right", style="bold")
    summary_table.add_row("Supported", f"[green]{supported}/{total}[/green]")
    summary_table.add_row("Weak support", f"[yellow]{weak}/{total}[/yellow]")
    summary_table.add_row("Contradictions", f"[red]{contradicts}/{total}[/red]")
    summary_table.add_row("Hallucinations", f"[bold red]{hallucinated}/{total}[/bold red]")
    summary_table.add_row("Unverifiable denials", f"[magenta]{denials}/{total}[/magenta]")
    console.print(summary_table)
    console.print()

# Claim detail
    bad_claims = [c for c in claims if c["status"] in ("CONTRADICTION", "HALLUCINATION")]
    uncertain_claims = [c for c in claims if c["status"] == "UNVERIFIABLE_DENIAL"]
    ok_claims = [c for c in claims if c["status"] in ("SUPPORTED", "WEAK_SUPPORT")]

    if ok_claims and Confirm.ask("[dim]Show supported claims too?[/dim]", default=False):
        console.print(Rule("[bold]Supported Claims[/bold]", style="dim green"))
        console.print()
        for c in ok_claims:
            colour = "green" if c["status"] == "SUPPORTED" else "yellow"
            label = "SUPPORTED" if c["status"] == "SUPPORTED" else "WEAK"
            console.print(f"  [{colour}]{label}[/{colour}]  {c.get('claim','')}")
            if c.get("unsupported_terms"):
                console.print(f"  [dim]unsupported terms: {', '.join(c['unsupported_terms'])}[/dim]")
        console.print()

    if uncertain_claims:
        console.print(Rule("[bold]Needs Review[/bold]", style="magenta"))
        console.print()
        for c in uncertain_claims:
            console.print(Panel(
                f"[magenta]{c['status']}[/magenta]  [dim]score {round(c.get('score', 0), 3)}[/dim]\n\n"
                f"{c.get('claim', '')}\n\n"
                + (f"[dim]Unsupported terms:[/dim] {', '.join(c['unsupported_terms'])}\n"
                   if c.get("unsupported_terms") else "")
                + (f"\n[dim]Closest chunk ({c.get('matched_source','')}, "
                   f"chunk {c.get('matched_chunk_id','')}):[/dim]\n{c.get('chunk_text','')}"
                   if c.get("chunk_text") else ""),
                border_style="magenta",
                padding=(1, 2),
            ))
            console.print()

    if bad_claims:
        console.print(Rule("[bold]Issues[/bold]", style="red"))
        console.print()
        for c in bad_claims:
            colour = "red" if c["status"] == "HALLUCINATION" else "bold red"
            console.print(Panel(
                f"[{colour}]{c['status']}[/{colour}]  [dim]score {round(c.get('score', 0), 3)}[/dim]\n\n"
                f"{c.get('claim', '')}\n\n"
                + (f"[dim]Reason:[/dim] {c.get('reason', '')}\n" if c.get('reason') else "")
                + (f"[dim]AI numbers:[/dim] {c.get('ai_numbers','')}  "
                   f"[dim]Truth numbers:[/dim] {c.get('truth_numbers','')}\n"
                   if c.get("ai_numbers") else "")
                + (f"[dim]Unsupported terms:[/dim] {', '.join(c['unsupported_terms'])}\n"
                   if c.get("unsupported_terms") else "")
                + (f"\n[dim]Closest chunk ({c.get('matched_source','')}, "
                   f"chunk {c.get('matched_chunk_id','')}):[/dim]\n{c.get('chunk_text','')}"
                   if c.get("chunk_text") else ""),
                border_style="red",
                padding=(1, 2),
            ))
            console.print()
    else:
        console.print("[bold green]✓ No hallucinations or contradictions found.[/bold green]")
        console.print()


def main():
    config = collect_inputs()
    source_docs, ai_output, verification = run_engine(config)
    render_results(source_docs, ai_output, verification, config)


if __name__ == "__main__":
    main()
