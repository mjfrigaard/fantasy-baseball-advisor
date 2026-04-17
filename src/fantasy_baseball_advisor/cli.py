"""CLI entry points via Click."""

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
@click.version_option()
def cli():
    """Fantasy Baseball Advisor — get data-driven roster insights."""


@cli.command()
@click.argument("player")
@click.option("--season", default=2024, show_default=True, help="MLB season year.")
def stats(player: str, season: int):
    """Show hitting stats for PLAYER."""
    click.echo(f"Fetching stats for {player} ({season})...")
