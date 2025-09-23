#!/bin/bash
for i in {0..10}
do
    python analyze_single_session.py "$i"
done
