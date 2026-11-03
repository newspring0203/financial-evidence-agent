"""MCP stdio adapter over the same deterministic domain tools.
Run collection explicitly before starting. No API keys in tool arguments.
"""
import os
from mcp.server.fastmcp import FastMCP
from .domain import Tools

mcp = FastMCP('Financial Evidence Agent')


def tools():
    return Tools(os.getenv('FEA_MCP_SOURCE', 'dart'), 'bm25')


@mcp.tool()
def get_financials(company: str, year: int, basis: str, metric: str) -> dict:
    """Retrieve annual financial fact, currency, basis and official receipt URL."""
    return tools().get_financials(company, year, basis, metric)


@mcp.tool()
def calculate_change(current_id: str, prior_id: str) -> dict:
    """Calculate consecutive annual percentage change using existing fact IDs."""
    return tools().calculate_change(current_id, prior_id)


@mcp.tool()
def search_disclosures(company: str, year: int, basis: str, query: str) -> list:
    """Retrieve disclosure chunks with exact text offsets and source hashes."""
    return tools().search_disclosures(company, year, basis, query)


@mcp.tool()
def reconcile_prior(current_id: str, prior_id: str) -> dict:
    """Compare prior-year comparative in current report against previous annual report."""
    return tools().reconcile_prior(current_id, prior_id)


if __name__ == '__main__':
    mcp.run(transport='stdio')
