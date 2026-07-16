import os

from github import Github

from .mcp_instance import mcp


def _client() -> Github:
    return Github(os.environ["GITHUB_TOKEN"])


@mcp.tool()
def github_list_repos(org: str = "") -> list[str]:
    """List repositories for the authenticated user, or for an org if given"""
    gh = _client()
    target = gh.get_organization(org) if org else gh.get_user()
    return [r.full_name for r in target.get_repos()]


@mcp.tool()
def github_list_issues(repo: str, state: str = "open") -> list[dict]:
    """List issues for a repo (format 'owner/repo')"""
    repository = _client().get_repo(repo)
    return [
        {"number": i.number, "title": i.title, "state": i.state, "url": i.html_url}
        for i in repository.get_issues(state=state)
    ]


@mcp.tool()
def github_create_issue(repo: str, title: str, body: str = "") -> str:
    """Create an issue in a repo (format 'owner/repo'). Returns the issue URL."""
    repository = _client().get_repo(repo)
    issue = repository.create_issue(title=title, body=body)
    return issue.html_url


@mcp.tool()
def github_list_pull_requests(repo: str, state: str = "open") -> list[dict]:
    """List pull requests for a repo (format 'owner/repo')"""
    repository = _client().get_repo(repo)
    return [
        {"number": p.number, "title": p.title, "state": p.state, "url": p.html_url}
        for p in repository.get_pulls(state=state)
    ]


@mcp.tool()
def github_get_file_contents(repo: str, path: str, ref: str = "") -> str:
    """Read a file's contents from a repo (format 'owner/repo')"""
    repository = _client().get_repo(repo)
    kwargs = {"ref": ref} if ref else {}
    content_file = repository.get_contents(path, **kwargs)
    return content_file.decoded_content.decode("utf-8")
