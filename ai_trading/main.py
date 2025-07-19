"""Wrapper exposing the project main entry point."""
from importlib import import_module

# AI-AGENT-REF: delegate to root main
main = import_module("main").main

