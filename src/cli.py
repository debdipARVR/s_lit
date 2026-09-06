"""Command-line interface for ClozeCongruence AI Text Detector."""

from __future__ import annotations

import argparse
import json
import sys
from .engine import ClozeCongruenceDetector, TextHumanizer, NvidiaNIMClient
from .security.cli import main as security_main


def main():
    parser = argparse.ArgumentParser(
        description="ClozeCongruence - AI Text Detection & Humanization Engine"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. detect command
    det_parser = subparsers.add_parser("detect", help="Run cloze congruence AI detection on text")
    det_parser.add_argument("text", nargs="?", default="", help="Text string to evaluate")
    det_parser.add_argument("--file", "-f", type=str, help="Path to text file to evaluate")
    det_parser.add_argument("--mask-rate", "-m", type=float, default=0.30, help="Cloze mask rate (0.10 - 0.50)")
    det_parser.add_argument("--passes", "-p", type=int, default=2, help="Number of Monte Carlo passes")
    det_parser.add_argument("--model", type=str, default="z-ai/glm-5.2", help="NVIDIA NIM model ID")
    det_parser.add_argument("--json", action="store_true", help="Output full JSON results")

    # 2. humanize command
    hum_parser = subparsers.add_parser("humanize", help="Rewrite text with anti-detection styling and burstiness")
    hum_parser.add_argument("text", nargs="?", default="", help="Text to humanize")
    hum_parser.add_argument("--file", "-f", type=str, help="Path to file to humanize")
    hum_parser.add_argument("--domain", "-d", default="academic", choices=["academic", "conversational", "technical", "creative", "business"])

    # 3. prompt command
    p_parser = subparsers.add_parser("prompt", help="Generate anti-detection prompt template")
    p_parser.add_argument("--domain", "-d", default="academic")
    p_parser.add_argument("--audience", "-a", default="General Readership")

    # 4. serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI playground server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    # 5. security command
    subparsers.add_parser("security", help="Run Fernet key encryption tool")

    args, unknown = parser.parse_known_args()

    if args.subcommand == "detect":
        content = args.text
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()

        if not content.strip():
            print("Error: Please provide text or a file to analyze.", file=sys.stderr)
            sys.exit(1)

        detector = ClozeCongruenceDetector()
        res = detector.analyze(
            text=content,
            mask_rate=args.mask_rate,
            num_passes=args.passes,
            model_name=args.model,
        )

        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n=================================================================")
            print(" Two-Pass Cloze Congruence AI Detection Result (DeepEval)")
            print("=================================================================")
            print(f"Verdict:                    {res['verdict']} (Confidence: {res['confidence']})")
            print(f"Combined AI Probability:    {res['ai_probability']}%")
            print(f"Combined Congruence Score:  {res['combined_congruence_score']}%\n")
            
            p1 = res.get("pass_1", {})
            print(f"--- Pass 1 (Sparse: 1 sentence per 4 lines) ---")
            print(f"  Congruence Score:         {p1.get('congruence_score', 0)}%")
            print(f"  Meaning Similarity (55%): {p1.get('meaning_similarity', 0)}%")
            print(f"  Semantic Cosine (30%):    {p1.get('semantic_cosine', 0)}%")
            print(f"  DeepEval Reason:          {p1.get('deepeval_reason', 'N/A')}\n")

            p2 = res.get("pass_2", {})
            print(f"--- Pass 2 (Alternate: sentences every 2 lines) ---")
            print(f"  Congruence Score:         {p2.get('congruence_score', 0)}%")
            print(f"  Meaning Similarity (55%): {p2.get('meaning_similarity', 0)}%")
            print(f"  Semantic Cosine (30%):    {p2.get('semantic_cosine', 0)}%")
            print(f"  DeepEval Reason:          {p2.get('deepeval_reason', 'N/A')}\n")

            print(f"Two-Pass Verdict Reason:    {res.get('two_pass_verdict_reason', '')}")
            print("=================================================================\n")

    elif args.subcommand == "humanize":
        content = args.text
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()

        if not content.strip():
            print("Error: Please provide text or a file to humanize.", file=sys.stderr)
            sys.exit(1)

        humanizer = TextHumanizer()
        res = humanizer.humanize(content, domain=args.domain)
        print("\n=======================================================")
        print(f" Humanized Output ({args.domain.title()} Profile)")
        print("=======================================================")
        print(res["humanized_text"])
        print("=======================================================\n")

    elif args.subcommand == "prompt":
        humanizer = TextHumanizer()
        bundle = humanizer.generate_humanize_prompt(domain=args.domain, target_audience=args.audience)
        print(bundle["full_prompt"])

    elif args.subcommand == "serve":
        import uvicorn
        uvicorn.run("src.web.app:app", host=args.host, port=args.port)

    elif args.subcommand == "security":
        sys.argv = [sys.argv[0]] + unknown
        security_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
