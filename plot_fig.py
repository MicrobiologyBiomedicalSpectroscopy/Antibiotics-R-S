import matplotlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def mean_stb_data(X, beta=0.5):
    """
    This function calculate the confidence interval
    :param X: data (type: numpy array)
    :param beta: the width of the interval (type: float)
    :return: new data with mean, up_interval and low_interval
    """
    X_mean = X.mean()
    X_std = X.std()

    X_up = X_mean + beta * X_std
    X_low = X_mean - beta * X_std
    df = pd.DataFrame(X_mean, columns=["mean"])
    df["std"] = X_std
    df["X_up"] = X_up
    df["X_low"] = X_low

    return df, X_mean.values, X_up.values, X_low.values

def spectra_plot(df ,
                 label,
                 initial_feature="1801.264",
                 final_feature="898.703",
                 out_window_size_tuple_x = (900, 1800),
                 out_window_size_tuple_y = (-0.003, 0.255),
                 inset_axis=True,
                 inset_window_position=[0.02, 0.43, 0.75, 0.54],
                 labels_dict=[{"label": "S", "label_legend": "label_1", "color": "blue"}, {"label": "R", "label_legend": "label_2", "color": "red"}],
                 confidence_iterval_width=0.5,
                 texts_on_graph_dict=[{"which_axis": "ax2", "text_sentence": "ER", "position_x": 1000, "position_y": 0.025}],
                 functional_groups=True,
                 functional_groups_dict=[{"group_name": 'Amid III', "arrow_position": (1241, 0.028), "xytext": (1241, 0.05)},
                                         {"group_name": 'as CH' + r"$_3$", "arrow_position": (1456, 0.018), "xytext": (1456 - 8, 0.045)},
                                         {"group_name": 'Amid II', "arrow_position": (1548, 0.12), "xytext": (1548 - 27, 0.16)},
                                         {"group_name": 'Amid I', "arrow_position": (1659, 0.232), "xytext": (1659 - 25, 0.245)},
                                         {"group_name": "sym CCO" + r"$^-$", "arrow_position": (1400, 0.02), "xytext": (1400, 0.058)}],
                 inset_control_x=(958, 1180),
                 inset_control_y=(0, 0.1),
                 legend_location=2,
                 graph_item="none",
                 save_fig=False,
                 y_title="Absorbance (A.U)",
                 x_title="Wavenumber (cm" + r"$^{-1})$",
                 file_name="spectra"):
    """

    :param df: data frame (type: pandas df)
    :param label: name of column of the labels (type: string)
    :param initial_feature: first column name (type: string)
    :param final_feature: last column name(type: string)
    :param inset_axis: insert inset_axis as small window, True or False (type: boolean)
    :param inset_window_position: the limits of the inset window on respect to the major window (type: list of floats)
    :param labels_dict: list of dictionaries include the details about the classes (type: list of dictionaries)
    :param confidence_iterval_width: the width of the confidence interval (type: float)
    :param texts_on_graph_dict: list of dictionaries include the details about texts on the graph (type: list of dictionaries)
    :param functional_groups: insert functional groups as text-arrows, True or False (type: boolean)
    :param functional_groups_dict: list of dictionaries include the details about vibration functional groups (type: list of dictionaries)
    :param inset_control_x: tuning the x-axis limits inside the inset-window (type: tuple)
    :param inset_control_y: tuning the y-axis limits inside the inset-window (type: tuple)
    :param legend_location: legend location on the window (type: integer)
    :param graph_item: item for the figure, default "none" don't add any thing (type: string)
    :param save_fig: do you want to save the figure?True or False (type: boolean)
    :param y_title: title of the y-axis (type: string)
    :param x_title: title of the x-axis (type: string)
    :param file_name: name of the file in case of saving the figure in file (type: string)

    :return: show the spectra figure
    """
    wavenumbers = df.loc[:, initial_feature: final_feature].columns.values.astype("float")
    X = df.loc[:, initial_feature: final_feature]
    y = df[label]

    df_sub = pd.concat([X, y], axis=1)
    df_sub = df_sub.dropna()

    plt.rc("font", size=16, family="Times New Roman")
    plt.rc('axes', linewidth=2)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_axes([0, 0, 1, 1])

    if inset_axis:
        ax2 = fig.add_axes(inset_window_position)
        ax2.set_yticks([])
        for lab_indx in range(len(labels_dict)):
            spectra = df_sub[df_sub[label] == labels_dict[lab_indx]["label"]]
            spectra = pd.DataFrame(spectra.drop(label, axis=1).values.astype("float"))
            _, spec_mean, spec_up, spec_low = mean_stb_data(spectra, beta=confidence_iterval_width)
            ax1.plot(wavenumbers, spec_mean, color=labels_dict[lab_indx]["color"])
            ax1.fill_between(wavenumbers, spec_up, spec_low, color=labels_dict[lab_indx]["color"], alpha=0.1)

            ax2.plot(wavenumbers, spec_mean, color=labels_dict[lab_indx]["color"], label=labels_dict[lab_indx]["label_legend"])
            ax2.fill_between(wavenumbers, spec_up, spec_low, color=labels_dict[lab_indx]["color"], alpha=0.1)

            ax2.set_xlim(inset_control_x)
            ax2.tick_params(axis='x', which='major', labelsize=12)
            ax2.set_ylim(inset_control_y)



    else:
        for lab_indx in range(len(labels_dict)):
            spectra = df_sub[df_sub[label] == labels_dict[lab_indx]["label"]]
            spectra = pd.DataFrame(spectra.drop(label, axis=1).values.astype("float"))
            _, spec_mean, spec_up, spec_low = mean_stb_data(spectra, beta=confidence_iterval_width)
            ax1.plot(wavenumbers, spec_mean, color=labels_dict[lab_indx]["color"], label=labels_dict[lab_indx]["label_legend"])
            ax1.fill_between(wavenumbers, spec_up, spec_low, color=labels_dict[lab_indx]["color"], alpha=0.1)

    if functional_groups:
        for gr in functional_groups_dict:
            ax1.annotate(gr["group_name"], xy=gr["arrow_position"], xytext=gr["xytext"],
                         arrowprops=dict(facecolor='black', shrink=0.05, width=2))

    ax1.set_xlabel(x_title, fontdict=dict(size=23, fontname="Times New Roman", fontweight='bold'))
    ax1.set_ylabel(y_title, fontdict=dict(size=23, fontname="Times New Roman", fontweight='bold'))
    ax1.set_xlim(out_window_size_tuple_x)
    ax1.set_ylim(out_window_size_tuple_y)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    for tex in texts_on_graph_dict:
        if tex["which_axis"] == "ax2":
            ax2.text(tex["position_x"], tex["position_y"], s=tex["text_sentence"], fontdict=dict(size=22, fontname="Times New Roman"))
            ax2.legend(loc=legend_location, ncol=1)
        elif tex["which_axis"] == "ax1":
            ax1.text(tex["position_x"], tex["position_y"], s=tex["text_sentence"], fontdict=dict(size=22, fontname="Times New Roman"))
            ax1.legend(loc=legend_location, ncol=1)

    if graph_item != "none":
        ax1.text(1260, 0.0016, s= graph_item, fontdict=dict(size=25, fontname="Times New Roman", fontweight="bold", style="italic"))

    if save_fig:
        fig.savefig(file_name + texts_on_graph_dict[0]["text_sentenceju"] + ".pdf", dpi=600, bbox_inches='tight',pil_kwargs={"compression": "tiff_lzw"})