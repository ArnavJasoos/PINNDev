"""NiceGUI front-end: async transcript, clarification/approval interrupts, live plots.

``bridge`` holds the framework-free glue (initial-state construction and event→
transcript mapping) so it is testable without nicegui/langgraph; ``app`` and
``components`` build the actual UI and are imported only when the GUI is launched.
"""
