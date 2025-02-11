from matplotlib import pyplot as plt


def figure():
    fig = plt.figure(figsize=(3.2, 1.7))

    return fig


def subplots_adjust():
    plt.subplots_adjust(left=0.19, right=0.84, top=1, bottom=0.235)
