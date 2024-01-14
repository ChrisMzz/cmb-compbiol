@echo off
title sigma_k sigma_c sweep
color 07

:: before launching this file, we need to format the filename strings accordingly
(
    for %%k in (0.01, 0.05, 0.1, 0.5, 1, 5, 10) do (
        for %%c in (0.01, 0.05, 0.1, 0.5, 1, 5, 10) do (
            python individuals_testing.py sk_sc %%k %%c
        )
    )
) > dump/skscsweep_log.txt

color 0a
echo All Done!
