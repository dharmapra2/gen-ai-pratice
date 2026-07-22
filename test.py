import trafilatura

downloaded = trafilatura.fetch_url('https://www.mockexperts.com/blog/agentic-ai-engineer-interview-questions-salary-how-to-get-hired-2026')
text = trafilatura.extract(downloaded)
print(text)
