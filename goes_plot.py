import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from matplotlib.ticker import FixedLocator
import datetime, html, argparse
from sunpy.time import TimeRange, parse_time
from sunpy.timeseries import TimeSeries
from crispy.utils import pt_vibrant

def time_to_td(time):
    return datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)

def time_idx(ts, time):
    times = np.array([time_to_td(x.datetime.time()) for x in parse_time(ts.data.index)])

    time = datetime.time.fromisoformat(time)
    
    diffs = times - datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)
    
    for j, d in enumerate(diffs):
        if d.days == -1:
            pass
        else:
            return j

def goes_plot(f, save=False, output_file="goes.png"):
    if type(f) == str:
        f = TimeSeries(f)

    aa = html.unescape("&#8491;")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dates = date2num(parse_time(f.data.index).datetime)

    ax.plot_date(dates, f.data["xrsa"], "-", label=f"0.5-4.0{aa}", color=pt_vibrant["blue"], lw=2)
    ax.plot_date(dates, f.data["xrsb"], "-", label=f"1.0-8.0{aa}", color=pt_vibrant["red"], lw=2)

    ax.set_yscale("log")
    ax.set_ylim(1e-9,1e-3)
    ax.set_title("GOES-15 X-Ray Flux 2014/09/06")
    ax.set_ylabel(r"Flux [Wm$^{-2}$]")
    ax.set_xlabel(datetime.datetime.isoformat(f.data.index[1])[0:10])

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(1e-9,1e-3)
    label = ["A", "B", "C", "M", "X"]
    centres = np.logspace(-8, -4, len(label))
    ax2.yaxis.set_major_locator(FixedLocator(centres))
    ax2.set_yticklabels(label)
    ax2.set_yticklabels([], minor=True)

    ax.yaxis.grid(True, "major")
    ax.xaxis.grid(False, "major")

    ax.legend(loc="lower right")

    formatter = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)

    ax.fmt_xdata = DateFormatter('%H:%M')
    fig.autofmt_xdate()

    if save:
        fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    else:
        fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="The filename to plot.", type=str)
    parser.add_argument("-s","--save", help="Whether or not to save the file.", action="store_true")
    parser.add_argument("--output_file", help="The file to save to.", default="goes.png", type=str)
    args = parser.parse_args()

    goes_plot(args.file, args.save, args.output_file)