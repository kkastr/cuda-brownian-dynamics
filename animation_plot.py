import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as pg
from PIL import Image
from tqdm import tqdm


def main():

    if not os.path.exists("images"):
        os.mkdir("images")

    fname = "./data/trajectory-N10000-frames1000-lbox1000-seed1234.csv"

    N, nframes, lbox, seed = (int(num) for num in re.findall(r"\d+", fname))

    outfreq = 100
    dt = 0.1

    df = pd.read_csv(fname)

    dfn = df.astype(
        {"frame": np.int16, "id": np.int16, "x": np.float32, "y": np.float32, "z": np.float32}
    )

    lo, hi = 1500, 2000
    rdf = dfn[(dfn.id >= lo) & (df.id < hi)].reset_index()

    num = hi - lo

    flo, fhi = 200, 1000

    nframes = fhi - flo

    start = flo * num
    stop = fhi * num
    dfa = rdf.loc[start:stop, :]

    xmin, ymin, zmin = np.round(dfa[["x", "y", "z"]].min())

    xmax, ymax, zmax = np.round(dfa[["x", "y", "z"]].max())

    xrange = dict(range=[xmin, xmax], autorange=False, visible=False)
    yrange = dict(range=[ymin, ymax], autorange=False, visible=False)
    zrange = dict(range=[zmin, zmax], autorange=False, visible=False)

    camera = dict(projection=dict(type="orthographic"))

    scene = dict(camera=camera, xaxis=xrange, yaxis=yrange, zaxis=zrange)

    line_style = dict(width=2, color="DarkSlateGrey")

    marker_style = dict(size=6, color="crimson", line=line_style)

    ## notebook interactive figure

    # play_args = [
    #     None,
    #     dict(
    #         frame=dict(duration=10, redraw=True),
    #         fromcurrent=True,
    #         transition=dict(duration=30, easing="linear"),
    #     ),
    # ]

    # # play_args = [None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)]

    # play_button = dict(label="Play", method="animate", args=play_args)

    # pause_args = [
    #     [None],
    #     dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0)),
    # ]

    # pause_button = dict(label="Pause", method="animate", args=pause_args)

    # menus = dict(
    #     type="buttons",
    #     buttons=[play_button, pause_button],
    #     direction="left",
    #     pad=dict(r=10, t=87),
    #     showactive=False,
    #     x=0.1,
    #     xanchor="right",
    #     y=0,
    #     yanchor="top",
    # )

    # menus = dict(type='buttons', buttons=[play_button, pause_button])

    sliders = dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(font=dict(size=20), prefix="Frame", visible=True, xanchor="right"),
        transition=dict(duration=30, easing="linear"),
        pad=dict(b=10, t=87),
        len=0.9,
        x=0.1,
        y=0,
        steps=[],
    )

    setup = [pg.Scatter3d(x=[], y=[], z=[], mode="markers", marker=marker_style)]

    step = 5

    fid = dfa.frame.unique()
    fkeys = [fid[i] - fid.min() for i in range(len(fid)) if i % step == 0]

    fx = dfa.x.values.reshape((nframes, num))
    fy = dfa.y.values.reshape((nframes, num))
    fz = dfa.z.values.reshape((nframes, num))

    slider_args = dict(
        frame=dict(duration=0, redraw=True),
        fromcurrent=True,
        mode="immediate",
        transition=dict(duration=0),
    )
    slider_steps = [
        dict(args=[[f"frame{i}"], slider_args], label=f"{i}", method="animate") for i in fkeys[1:]
    ]

    sliders["steps"] = slider_steps

    fig = pg.Figure(data=setup)

    zoom_factor = 0.5

    xzoom, yzoom, zzoom = zoom_factor, zoom_factor, zoom_factor
    camera = dict(eye=dict(x=xzoom, y=yzoom, z=zzoom))
    scene = dict(camera=camera, xaxis=xrange, yaxis=yrange, zaxis=zrange)

    fig.update_layout(scene=scene)
    fig.update_layout(paper_bgcolor="whitesmoke")
    fig.update_layout(plot_bgcolor="whitesmoke")

    fig.update_layout(showlegend=False)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    imglst = []

    frames = [
        pg.Frame(
            data=[pg.Scatter3d(x=fx[k, :], y=fy[k, :], z=fz[k, :])], traces=[0], name=f"frame{k}"
        )
        for k in fkeys
    ]

    for k in tqdm(fkeys):
        idx = k // step
        outfile = f"images/fig{idx}.png"

        xyz = [pg.Scatter3d(x=fx[k, :], y=fy[k, :], z=fz[k, :])]
        fig.update(data=xyz)
        fig.write_image(outfile)

        # imglst.append(Image.open(outfile))


if __name__ == "__main__":
    main()
