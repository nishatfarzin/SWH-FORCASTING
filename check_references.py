"""Check every reference against Crossref (run on your own computer with internet access).

Input : references.csv with columns key,doi,first_author,year,title
Output: reference_check.csv with the Crossref title, first author, year, and a status per row.

A row is "OK" when the DOI resolves, the year matches and the title is close to the Crossref title.
Everything else needs a manual look. Use a contact e-mail in the User-Agent (Crossref asks for it).
"""

import argparse
import csv
import difflib
import time
import urllib.parse

import requests


def norm(text):
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch == " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="references.csv")
    ap.add_argument("--output", default="reference_check.csv")
    ap.add_argument("--email", required=True, help="contact address for the Crossref polite pool")
    args = ap.parse_args()

    headers = {"User-Agent": f"reference-check/1.0 (mailto:{args.email})"}
    out_rows = []
    with open(args.input, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            doi = row["doi"].strip()
            status, cr_title, cr_first, cr_year = "", "", "", ""
            url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    status = f"DOI NOT FOUND (HTTP {resp.status_code})"
                else:
                    msg = resp.json()["message"]
                    cr_title = (msg.get("title") or [""])[0]
                    authors = msg.get("author") or [{}]
                    cr_first = authors[0].get("family", "")
                    issued = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
                    cr_year = str(issued or "")
                    ratio = difflib.SequenceMatcher(None, norm(row["title"]), norm(cr_title)).ratio()
                    problems = []
                    if cr_year and cr_year != row["year"].strip():
                        problems.append(f"year differs ({row['year']} vs {cr_year})")
                    if ratio < 0.85:
                        problems.append(f"title differs (similarity {ratio:.2f})")
                    if (
                        cr_first
                        and norm(cr_first) not in norm(row["first_author"])
                        and norm(row["first_author"]) not in norm(cr_first)
                    ):
                        problems.append("first author differs")
                    status = "OK" if not problems else "CHECK: " + "; ".join(problems)
            except requests.RequestException as exc:
                status = f"NETWORK ERROR: {exc}"
            out_rows.append(
                {
                    **row,
                    "crossref_title": cr_title,
                    "crossref_first_author": cr_first,
                    "crossref_year": cr_year,
                    "status": status,
                }
            )
            time.sleep(0.2)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    bad = [r for r in out_rows if r["status"] != "OK"]
    print(f"{len(out_rows)} references checked, {len(bad)} need attention -> {args.output}")


if __name__ == "__main__":
    main()
