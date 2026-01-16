# # save_plasma_colorbar.py
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np

# def save_colorbar_pdf(filename=“plasma_colorbar.pdf”,
#                         cmap_name=“plasma”,
#                         orientation=“horizontal”,  # “horizontal” or “vertical”
#                         size=(6, 1.0),            # figsize in inches for horizontal
#                         fontsize=12,
#                         vmin=0.0, vmax=1.0):
#         “”"
#         Save only the colorbar for the given colormap to a PDF file.
#         orientation: “horizontal” or “vertical”
#         size: figure size (width, height) in inches (adjust depending on orientation)
#         “”"
#         # Normalize for the colorbar
#         norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#         cmap = plt.get_cmap(cmap_name)
#         # create figure with a single axes reserved for the colorbar
#         fig = plt.figure(figsize=size)
#         if orientation == “horizontal”:
#             # axes: [left, bottom, width, height] in fraction of figure
#             cax = fig.add_axes([0.05, 0.25, 0.9, 0.5])
#         else:  # vertical
#             # a tall narrow figure works better for vertical
#             cax = fig.add_axes([0.3, 0.05, 0.2, 0.9])
#         # Create ColorbarBase (colorbar with no attached mappable)
#         cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orientation)
#         cb.set_label(f”Colormap: {cmap_name}“, fontsize=fontsize)
#         cb.ax.tick_params(labelsize=fontsize * 0.9)
#         # Save as PDF
#         fig.savefig(filename, bbox_inches=“tight”, pad_inches=0.02)
#         plt.close(fig)
#         print(f”Saved colorbar to: {filename}“)

# if __name__ == “__main__“:
#     # 横向きカラーバーを保存（ファイル名と向きを必要に応じて変更）
#     save_colorbar_pdf(filename=“plasma_colorbar_horizontal.pdf”,
#                       cmap_name=“plasma”,
#                       orientation=“horizontal”,
#                       size=(6, 1.0))
#     # 縦向きカラーバーも保存する例（コメントアウトを外して使ってください）
#     # save_colorbar_pdf(filename=“plasma_colorbar_vertical.pdf”,
#     #                   cmap_name=“plasma”,
#     #                   orientation=“vertical”,
#     #                   size=(1.0, 6))



# from colorspacious import cspace_converte

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list
    
    plt.show()


plot_color_gradients('Perceptually Uniform Sequential',
                     ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])