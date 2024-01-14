@echo off
title x0 sigma_k sweep
color 07

:: before launching this file, we need to format the filename strings accordingly
(
    for %%x in (-2, -1, 0, 1, 2) do (
        for %%k in (0.01, 0.05, 0.1, 0.5, 1, 5, 10) do (
            python individuals_testing.py x0_sk %%x %%k
        )
    )
) > dump/x0sksweep_log.txt

color 0a
echo All Done!
pause
