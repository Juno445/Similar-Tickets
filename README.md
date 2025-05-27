Certainly! Here is the previous professional README content as plain Markdown (no fenced code block wrapping):

# Duplicate Ticket Finder

Duplicate Ticket Finder is a Dockerized Python application that fetches tickets from a Freshservice instance, detects potential duplicates using powerful ML-based text embeddings (via [sentence-transformers](https://www.sbert.net/)), and outputs a CSV of likely duplicates for manual review. It is designed for automated, production deployment (e.g., scheduled weekly via Docker and Task Scheduler).

---

## Features

- **Automated Ticket Fetch:** Connect to your Freshservice API to pull tickets based on customizable queries.
- **ML-Powered Deduplication:** Uses modern NLP embeddings to compare subjects/descriptions, robust to wording changes and synonyms.
- **Highly Configurable:** Adjust similarity thresholds, filter strategies, and weighting.
- **Easy CSV Review:** Outputs well-structured CSV in a host-mounted folder.
- **Production-Ready:** Fully Dockerized for reliable, scheduled operation on Windows or Linux.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Automation (Scheduling)](#automation-scheduling)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

- **Docker** (recommended) – [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- _or:_ **Python 3.8+** (if running locally)
- Access credentials for your Freshservice API

---

## Installation

**Clone the repository:**

```sh
git clone https://github.com/yourorg/duplicate-ticket-finder.git
cd duplicate-ticket-finder
```

---

### Quick Start with Docker

1. **Build the Docker image:**

   ```sh
   docker build -t ticket-dedup -f docker/Dockerfile .
   ```

2. **(Windows only) Make sure Docker Desktop has access to the drive/folder you’ll use for output.**

3. **Run the container and mount a local data folder:**

   For Windows, use forward slashes and correct drive letter:
   ```sh
   docker run --rm -v "/c/Your/Output/Folder:/app/data" ticket-dedup
   ```
   For other OS:
   ```sh
   docker run --rm -v "$PWD/data:/app/data" ticket-dedup
   ```

4. **After successful run, results will be in your data folder.**

---

### Manual Python Run (development)

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```sh
   python src/main.py
   ```

---

## Configuration

- **API keys, group filters, and deduplication parameters** are set in `src/main.py`.
- _For production_: Move secrets (API key, domain, etc.) to environment variables or `config.json`, and load them in the script.
- _Config options include:_
  - **embedding_model**: sentence-transformers model name
  - **similarity_threshold**: float (0 to 1, higher is stricter)
  - **subject_weight**: how much to weight ticket subject vs. description

---

## Usage

- The app will fetch, process, and save:
  - `tickets.csv` — all fetched tickets
  - `potential_duplicates.csv` — list of detected duplicates

- Both are written to `data/` (host’s mapped folder).
- Review `potential_duplicates.csv` in Excel or your preferred tool.

---

## Automation (Scheduling)

### Windows Task Scheduler

Automate with Windows Task Scheduler to run every Monday at a chosen time:

1. Open **Task Scheduler**.
2. Create a Basic Task:
   - Action: `Start a program`
   - Program/script: `docker`
   - Arguments:
     ```
     run --rm -v /c/Your/Output/Folder:/app/data ticket-dedup
     ```
3. Set trigger to weekly, on Mondays.

See [official guide](https://docs.docker.com/desktop/) for additional Docker Desktop tips on scheduling and file shares.

---

## Output

- **Output CSV files** are delivered to your chosen host folder.
- Each run overwrites (or appends, per configuration) with up-to-date duplicate ticket analysis.
- Logs print to stdout (viewable via Docker or redirect to file in Task Scheduler).

---

## Troubleshooting

- **No output CSV?** Double-check your Docker volume mapping with a test file; see troubleshooting section in README or issues.
- **API errors?** Check your Freshservice credentials and access.
- **Performance:** For very large ticket sets, consider increasing RAM/CPU for Docker, or tune batch sizes in the code.

---

## Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue

1. Fork the repo
2. Create your feature branch (`git checkout -b my-feature`)
3. Commit your changes
4. Push to the branch (`git push origin my-feature`)
5. Open a Pull Request

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/)
- [Freshservice API](https://api.freshservice.com/)
- [Docker](https://www.docker.com/)

---

Created by Bryan Weston.
