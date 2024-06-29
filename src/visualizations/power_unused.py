from datautils.time import datetime_linspace
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian


        # bandwidth = 300
        # print(f"Window size: {1 / bandwidth} s")
        # win_size = int(1 / bandwidth * ds.stats.sampling_rate)
        # avg_power300 = moving_average(power_inst, win_size)
        # avg_power300_db = 10 * np.log10(avg_power300)

        # bandwidth = 100
        # print(f"Window size: {1 / bandwidth} s")
        # win_size = int(1 / bandwidth * ds.stats.sampling_rate)
        # avg_power100 = moving_average(power_inst, win_size)
        # avg_power100_db = 10 * np.log10(avg_power100)

        # bandwidth = 50
        # print(f"Window size: {1 / bandwidth} s")
        # win_size = int(1 / bandwidth * ds.stats.sampling_rate)
        # avg_power50 = moving_average(power_inst, win_size)
        # avg_power50_db = 10 * np.log10(avg_power50)

        # db = False
        # if db:
        #     power_inst = power_inst_db
        #     avg_power300 = avg_power300_db
        #     avg_power100 = avg_power100_db
        #     avg_power50 = avg_power50_db
        #     title = "Power (dB re 1 $\\mu$Pa$^2$)"
        #     filename = "avg_pulse_power_db.png"
        # else:
        #     title = "Power ($\\mu$Pa$^2$)"
        #     filename = "avg_pulse_power.png"

        # ymax = max(
        #     np.max(power_inst),
        #     np.max(avg_power300),
        #     np.max(avg_power100),
        #     np.max(avg_power50),
        # )
        # ymin = ymax - 30 if db else 0

        # fig, axs = plt.subplots(4, 1)
        # fig.suptitle(title)

        # ax = axs[0]
        # ax.plot(power_inst, label="$P(t)$")
        # ax.set_ylim(ymin, ymax)
        # ax.set_xticks([])
        # ax.set_xlabel(None)
        # ax.set_ylabel("$P(t)$")

        # ax = axs[1]
        # ax.plot(avg_power300, label="$\\widebar{P}$ (300 Hz)")
        # ax.set_ylim(ymin, ymax)
        # ax.set_xticks([])
        # ax.set_xlabel(None)
        # ax.set_ylabel("$\\widebar{P}$ (300 Hz)")

        # ax = axs[2]
        # ax.plot(avg_power100, label="$\\widebar{P}$ (100 Hz)")
        # ax.set_ylim(ymin, ymax)
        # ax.set_xticks([])
        # ax.set_xlabel(None)
        # ax.set_ylabel("$\\widebar{P}$ (100 Hz)")

        # ax = axs[3]
        # ax.plot(avg_power50, label="$\\widebar{P}$ (50 Hz)")
        # ax.set_ylim(ymin, ymax)
        # ax.set_xlabel(f"Samples after {np.datetime_as_string(time_start_i, unit='s')}")
        # ax.set_ylabel("$\\widebar{P}$ (50 Hz)")
        
        # fig.savefig(Path(args.destination / filename), bbox_inches="tight", facecolor="w", dpi=300)


        # data = ds.waveform
        # # Do a moving average:
        # data = np.abs(np.convolve(data, np.ones(100)/100, mode='valid')) ** 2

        # # plt.plot(data)
        # # plt.show()

        # # break
        # win = gaussian(int(args.tperseg * ds.stats.sampling_rate), std=8, sym=True)
        # # plt.plot(win)
        # # plt.show()

        # SFT = ShortTimeFFT(win, hop=int((1 / 100) * ds.stats.sampling_rate), fs=ds.stats.sampling_rate, mfft=args.nfft, scale_to="magnitude")
        # Sx = SFT.stft(ds.waveform[:, 0]).sum(axis=0)
        # Sx = 10 * np.log10(np.abs(Sx))
        # Sx[Sx < -100] = np.nan

        # plt.hist(Sx, 200)
        # plt.show()

        # Sx = SFT.stft(ds.waveform[:, 0])
        # np.save("spec.npy", Sx)
        # return
        # tvec = datetime_linspace(time_start_i, time_end_i, len(Sx))

        # plt.plot(tvec, abs(Sx))
        # plt.imshow(abs(Sx), aspect='auto', origin='lower')
        # plt.show()

        # filename = (
        #     np.datetime_as_string(time_start_i, unit="s")
        #     .replace("-", "")
        #     .replace("T", "_")
        #     .replace(":", "")
        #     + "_"
        #     + args.title.replace("/", "_")
        # )
        # fig = plotting.plot_spectrograms(
        #     ds,
        #     title=title,
        #     tperseg=args.tperseg,
        #     overlap=args.overlap,
        #     nfft=args.nfft,
        #     fmin=args.fmin,
        #     fmax=args.fmax,
        #     vmin=args.vmin,
        #     vmax=args.vmax,
        # )
        # fig.savefig(
        #     Path(args.destination / filename), bbox_inches="tight", facecolor="w"
        # )
