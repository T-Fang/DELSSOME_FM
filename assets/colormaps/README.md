# Colormaps used by the original DELSSOME figures

Copied here so the project does not depend on other users' directories.

| File | Variable | Used for | Source |
|---|---|---|---|
| `corr_mat_colorscale.mat` | `rbmap2` | FC matrices | `CBIG_private/utilities/matlab/figure_utilities/corr_mat_colorscale.mat` (identical in ftian's and tzeng's CBIG_private) |
| `Rapaeh_color_table_FCD.mat` | `c3` (RGB; an alpha column of ones is appended) | FCD matrices | `/home/shaoshi.z/storage/MFM/script/Rapaeh_color_table_FCD.mat` |

Both are used as in tzeng `DELSSOME_plus/analysis/analysis_utils.py::plot_time_series`:
`imshow(matrix, cmap=ListedColormap(table))` with a colorbar. The original uses the default
color range; `scripts/plot_comparison.py` instead shares one range between the two
implementations' panels of each row, so they can be compared.
