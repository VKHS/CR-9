"""CROSS-MORSE artifact ingestion, numerical review and publication gates.

One module produces evidence: s14_builder computes the joint S14 confirmation
campaign from sealed lifecycle runs with the runtime's own confirm-stage code,
re-executing a contrast through cm_experiments where a run sealed none.  The
rest of the package ingests, reviews and gates evidence.  It never assigns DOIs.
"""
