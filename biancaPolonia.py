import streamlit as st
import scipy
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Função para filtro passa-baixa


def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Frequência de Nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# Título do app
st.set_page_config(layout="wide")
st.title("Análise de Dados: Interpolação, Detrend e Filtro Passa-Baixa")

col1, col2, col3 = st.columns(3)
with col1:
    # Carregar o arquivo de texto
    uploaded_file_kinem = st.file_uploader(
        "Escolha um arquivo da cinemática", type=["csv"])
    if uploaded_file_kinem:
        df = pd.read_csv(uploaded_file_kinem, sep=",", engine='python')
        disp_x = (df.iloc[:, 0].values)/1000
        disp_y = (df.iloc[:, 1].values)/1000
        disp_z = (df.iloc[:, 2].values)/1000

        with col2:
            # Carregar o arquivo de texto
            uploaded_file_acc = st.file_uploader(
                "Escolha um arquivo do acelerômetro do smartphone", type=["txt"])
            # Lê o arquivo como DataFrame
            if uploaded_file_acc:
                df = pd.read_csv(uploaded_file_acc, sep=";", engine='python')
                tempo = df.iloc[:, 0].values
                acc_x = df.iloc[:, 1].values
                acc_y = df.iloc[:, 2].values
                acc_z = df.iloc[:, 3].values
                with col3:
                    uploaded_file_gyro = st.file_uploader(
                        "Escolha o arquivo do giroscópio do smartphine", type=["txt"])
                    # Lê o arquivo como DataFrame
                    if uploaded_file_gyro:
                        df = pd.read_csv(uploaded_file_gyro,
                                         sep=";", engine='python')
                        tempo_gyro = df.iloc[:, 0].values
                        gyro_x = df.iloc[:, 1].values
                        gyro_y = df.iloc[:, 2].values
                        gyro_z = df.iloc[:, 3].values

                        # cinemática
                        original_fs_kinem = 100
                        new_fs_kinem = 100
                        cutoff_kinem = 2
                        time_original_kinem = np.arange(
                            0, len(disp_y)) / original_fs_kinem
                        with col1:
                            valor = st.slider("Ajustar o trigger da cinemática", min_value=0, max_value=len(
                                time_original_kinem), value=0)
                        time_original_kinem = time_original_kinem - \
                            time_original_kinem[valor]

                        dy_dx = np.diff(disp_z) / np.diff(time_original_kinem)
                        baseline = np.mean(dy_dx)
                        sd_baseline = np.std(dy_dx)
                        # Diferença entre tempos consecutivos

                        peaks, properties = find_peaks(-1*disp_y, height=-1)
                        onsets = []
                        offsets = []
                        for peak in peaks:
                            # Busca para trás: início da queda
                            for i in range(peak, 1, -1):
                                if disp_y[i] > disp_y[i-1]:
                                    onsets.append(i)
                                    break

                            # Busca para frente: fim da queda
                            for i in range(peak, len(disp_y)-1):
                                if disp_y[i] > disp_y[i+1]:
                                    offsets.append(i)
                                    break

                        standing_time = []
                        for values in onsets:
                            for idx, i in enumerate(disp_z[values:values+200]):
                                if i == max(disp_z[values:values+200]):
                                    standing_time.append(
                                        time_original_kinem[values+idx])
                                    break

                        sitting_time = []
                        for values in offsets:
                            for idx, i in enumerate(disp_z[values-400:values]):
                                if i == max(disp_z[values-400:values]):
                                    sitting_time.append(
                                        time_original_kinem[values-400+idx])
                                    break

                        # aceleração
                        new_fs = 100
                        interpf = scipy.interpolate.interp1d(tempo, acc_x)
                        time_ = np.arange(
                            start=tempo[0], stop=tempo[len(tempo)-1], step=10)
                        x_ = interpf(time_)
                        time_interpolated, acc_x_interpolated = time_/1000, x_
                        interpf = scipy.interpolate.interp1d(tempo, acc_y)
                        time_ = np.arange(
                            start=tempo[0], stop=tempo[len(tempo)-1], step=10)
                        y_ = interpf(time_)
                        time_interpolated, acc_y_interpolated = time_/1000, y_
                        interpf = scipy.interpolate.interp1d(tempo, acc_z)
                        time_ = np.arange(
                            start=tempo[0], stop=tempo[len(tempo)-1], step=10)
                        z_ = interpf(time_)
                        time_interpolated, acc_z_interpolated = time_/1000, z_

                        acc_x_detrended = detrend(acc_x_interpolated)
                        acc_y_detrended = detrend(acc_y_interpolated)
                        acc_z_detrended = detrend(acc_z_interpolated)

                        # Filtro passa-baixa (10 Hz)
                        cutoff = 4  # Frequência de corte
                        acc_x_filtered = low_pass_filter(
                            acc_x_detrended, cutoff, new_fs)
                        acc_y_filtered = low_pass_filter(
                            acc_y_detrended, cutoff, new_fs)
                        acc_z_filtered = low_pass_filter(
                            acc_z_detrended, cutoff, new_fs)
                        acc_norm_filtered = np.sqrt(
                            acc_x_filtered**2+acc_y_filtered**2+acc_z_filtered**2)
                        
                        with col2:
                            valor_acc = st.slider(
                                "Ajuste o trigger do acc", min_value=0, max_value=len(time_interpolated), value=0)
                            time_interpolated = time_interpolated - \
                                time_interpolated[valor_acc]

                        # giroscópio
                        interpf = scipy.interpolate.interp1d(
                            tempo_gyro, gyro_x)
                        time_ = np.arange(
                            start=tempo_gyro[0], stop=tempo_gyro[len(tempo_gyro)-1], step=10)
                        x_ = interpf(time_)
                        time_interpolated_gyro, gyro_x_interpolated = time_/1000, x_
                        interpf = scipy.interpolate.interp1d(
                            tempo_gyro, gyro_y)
                        time_ = np.arange(
                            start=tempo_gyro[0], stop=tempo_gyro[len(tempo_gyro)-1], step=10)
                        y_ = interpf(time_)
                        time_interpolated_gyro, gyro_y_interpolated = time_/1000, y_
                        interpf = scipy.interpolate.interp1d(
                            tempo_gyro, gyro_z)
                        time_ = np.arange(
                            start=tempo_gyro[0], stop=tempo_gyro[len(tempo_gyro)-1], step=10)
                        z_ = interpf(time_)
                        time_interpolated_gyro, gyro_z_interpolated = time_/1000, z_

                        # Detrend
                        gyro_x_detrended = detrend(gyro_x_interpolated)
                        gyro_y_detrended = detrend(gyro_y_interpolated)
                        gyro_z_detrended = detrend(gyro_z_interpolated)

                        # Filtro passa-baixa (10 Hz)
                        cutoff = 1.25  # Frequência de corte
                        gyro_x_filtered = low_pass_filter(
                            gyro_x_detrended, cutoff, new_fs)
                        gyro_y_filtered = low_pass_filter(
                            gyro_y_detrended, cutoff, new_fs)
                        gyro_z_filtered = low_pass_filter(
                            gyro_z_detrended, cutoff, new_fs)
                        gyro_norm_filtered = np.sqrt(
                            gyro_x_filtered**2+gyro_y_filtered**2+gyro_z_filtered**2)
                        if np.mean(acc_x) > np.mean(acc_y):
                            ml_acc = np.sqrt(acc_y_filtered**2)
                            v_acc = np.sqrt(acc_x_filtered**2)
                            ap_acc = np.sqrt(acc_z_filtered**2)
                            ml_gyro = np.sqrt(gyro_y_filtered**2)
                            v_gyro = np.sqrt(gyro_x_filtered**2)
                            ap_gyro = np.sqrt(gyro_z_filtered**2)
                        else:
                            v_acc = np.sqrt(acc_y_filtered**2)
                            ml_acc = np.sqrt(acc_x_filtered**2)
                            ap_acc = np.sqrt(acc_z_filtered**2)
                            ml_gyro = np.sqrt(gyro_x_filtered**2)
                            v_gyro = np.sqrt(gyro_y_filtered**2)
                            ap_gyro = np.sqrt(gyro_z_filtered**2)
                        with col3:
                            valor_gyro = st.slider(
                                "Ajuste o trigger do gyro", min_value=0, max_value=len(time_interpolated_gyro), value=0)
                            time_interpolated_gyro = time_interpolated_gyro - \
                                time_interpolated_gyro[valor_gyro]

                        with col1:

                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(
                                time_original_kinem[0:2000], disp_z[0:2000], 'k-')
                            ax.plot([0, 0], [0, 2], 'r-')

                            num_ciclos = min(len(onsets), len(offsets))

                            # for i in range(num_ciclos):
                            #    t_onset = time_original_kinem[onsets[i]]
                            #    t_offset = time_original_kinem[offsets[i]]

                            #    # Linha tracejada: início
                            #    ax.axvline(t_onset, linestyle='--', color='orange',
                            #               label='Início da queda' if i == 0 else "")
                            # Linha tracejada: fim
                            #    ax.axvline(t_offset, linestyle='--', color='green',
                            #               label='Fim da queda' if i == 0 else "")
                            # Faixa entre onset e offset
                            #    ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                            # label='Fase de queda' if i == 0 else "")

                            #    # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                            #    if i + 1 < num_ciclos:
                            #        t_next_onset = time_original_kinem[onsets[i+1]]
                            #        ax.axvspan(t_offset, t_next_onset, color='lightblue',
                            #                   alpha=0.3, label='Intervalo' if i == 0 else "")

                            # Mínimos detectados
                            # for i, t in enumerate(time_original_kinem[peaks]):
                            #    ax.axvline(t, linestyle='--', color='blue',
                            # label='Mínimo' if i == 0 else "")

                            ax.set_xlabel("Tempo (s)")
                            ax.set_ylabel("Amplitude")
                            st.pyplot(fig)

                            with col2:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(
                                    time_interpolated[0:2000], acc_norm_filtered[0:2000], 'k-')

                                ax.plot([0, 0], [0, 30], 'r--')
                                # Verificação básica para evitar erros
                                # num_ciclos = min(len(onsets), len(offsets))

                                # for i in range(num_ciclos):
                                #    t_onset = time_original_kinem[onsets[i]]
                                #    t_offset = time_original_kinem[offsets[i]]

                                #    # Linha tracejada: início
                                #    ax.axvline(t_onset, linestyle='--', color='orange',
                                #               label='Início da queda' if i == 0 else "")
                                #    # Linha tracejada: fim
                                #    ax.axvline(t_offset, linestyle='--', color='green',
                                #               label='Fim da queda' if i == 0 else "")
                                #    # Faixa entre onset e offset
                                #    ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                #               label='Fase de queda' if i == 0 else "")

                                # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                #    if i + 1 < num_ciclos:
                                #        t_next_onset = time_original_kinem[onsets[i+1]]
                                #        ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                #                   alpha=0.3, label='Intervalo' if i == 0 else "")

                                # Mínimos detectados
                                # for i, t in enumerate(time_original_kinem[peaks]):
                                #    ax.axvline(t, linestyle='--', color='blue',
                                #               label='Mínimo' if i == 0 else "")
                                ax.set_xlabel("Tempo (s)")
                                ax.set_ylabel("Amplitude")
                                st.pyplot(fig)
                            with col3:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(
                                    time_interpolated_gyro[0:2000],  gyro_norm_filtered[0:2000], 'k-')
                                ax.plot([0, 0], [0, 2], 'r-')
                                # Verificação básica para evitar erros
                                # num_ciclos = min(len(onsets), len(offsets))

                                # for i in range(num_ciclos):
                                #    t_onset = time_original_kinem[onsets[i]]
                                #    t_offset = time_original_kinem[offsets[i]]

                                # Linha tracejada: início
                                #    ax.axvline(t_onset, linestyle='--', color='orange',
                                #               label='Início da queda' if i == 0 else "")
                                #    # Linha tracejada: fim
                                #    ax.axvline(t_offset, linestyle='--', color='green',
                                #               label='Fim da queda' if i == 0 else "")
                                #    # Faixa entre onset e offset
                                #    ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                #               label='Fase de queda' if i == 0 else "")

                                # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                #    if i + 1 < num_ciclos:
                                #        t_next_onset = time_original_kinem[onsets[i+1]]
                                #        ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                #                   alpha=0.3, label='Intervalo' if i == 0 else "")

                                # Mínimos detectados
                                # for i, t in enumerate(time_original_kinem[peaks]):
                                #    ax.axvline(t, linestyle='--', color='blue',
                                #               label='Mínimo' if i == 0 else "")
                                ax.set_xlabel("Tempo (s)")
                                ax.set_ylabel("Amplitude")
                                st.pyplot(fig)

                                with col1:
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    ax.plot(time_original_kinem, disp_y, 'k-')
                                    # Verificação básica para evitar erros
                                    num_ciclos = min(len(onsets), len(offsets))
                                    for i in range(num_ciclos):
                                        t_onset = time_original_kinem[onsets[i]]
                                        t_offset = time_original_kinem[offsets[i]]
                                    # Linha tracejada: início
                                        ax.axvline(t_onset, linestyle='--', color='orange',
                                                   label='Início da queda' if i == 0 else "")
                                    # Linha tracejada: fim
                                        ax.axvline(t_offset, linestyle='--', color='green',
                                                   label='Fim da queda' if i == 0 else "")
                                    # Faixa entre onset e offset
                                        ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                   label='Fase de queda' if i == 0 else "")

                                    # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                        if i + 1 < num_ciclos:
                                            t_next_onset = time_original_kinem[onsets[i+1]]
                                            ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                       alpha=0.3, label='Intervalo' if i == 0 else "")

                                    # Mínimos detectados
                                    for i, t in enumerate(time_original_kinem[peaks]):
                                        ax.axvline(t, linestyle='--', color='blue',
                                                   label='Mínimo' if i == 0 else "")

                                    ax.set_xlabel("Tempo (s)")
                                    ax.set_ylabel("Amplitude")

                                    st.pyplot(fig)
                                    with col2:
                                        fig, ax = plt.subplots(figsize=(10, 4))
                                        ax.plot(
                                            time_original_kinem, disp_z, 'k-')
                                        ax.plot([0, 0], [0, 2], 'r-')

                                        # Verificação básica para evitar erros

                                        num_ciclos = min(
                                            len(onsets), len(offsets))

                                        for i in range(num_ciclos):
                                            t_onset = time_original_kinem[onsets[i]]
                                            t_offset = time_original_kinem[offsets[i]]

                                            #    # Linha tracejada: início
                                            ax.axvline(t_onset, linestyle='--', color='orange',
                                                       label='Início da queda' if i == 0 else "")
                                            # Linha tracejada: fim
                                            ax.axvline(t_offset, linestyle='--', color='green',
                                                       label='Fim da queda' if i == 0 else "")
                                            # Faixa entre onset e offset
                                            ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                       label='Fase de queda' if i == 0 else "")
                                            #    # Linha tracejada: início
                                            ax.axvline(standing_time[i], linestyle='--', color='red',
                                                       label='Início da queda' if i == 0 else "")
                                            ax.axvline(sitting_time[i], linestyle='--', color='black',
                                                       label='Início da queda' if i == 0 else "")

                                            # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                            if i + 1 < num_ciclos:
                                                t_next_onset = time_original_kinem[onsets[i+1]]
                                                ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                           alpha=0.3, label='Intervalo' if i == 0 else "")

                                        # Mínimos detectados
                                        for i, t in enumerate(time_original_kinem[peaks]):
                                            ax.axvline(t, linestyle='--', color='blue',
                                                       label='Mínimo' if i == 0 else "")

                                        ax.set_xlabel("Tempo (s)")
                                        ax.set_ylabel("Amplitude")

                                        st.pyplot(fig)
                                        with col3:
                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, acc_norm_filtered, 'k-')
                                            ax.plot([0, 0], [0, 30], 'r--')
                                            # Verificação básica para evitar erros
                                            num_ciclos = min(
                                                len(onsets), len(offsets))

                                            for i in range(num_ciclos):
                                                t_onset = time_original_kinem[onsets[i]]
                                                t_offset = time_original_kinem[offsets[i]]

                                                # Linha tracejada: início
                                                ax.axvline(t_onset, linestyle='--', color='orange',
                                                           label='Início da queda' if i == 0 else "")
                                                # Linha tracejada: fim
                                                ax.axvline(t_offset, linestyle='--', color='green',
                                                           label='Fim da queda' if i == 0 else "")
                                                # Faixa entre onset e offset
                                                ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                           label='Fase de queda' if i == 0 else "")

                                                ax.axvline(standing_time[i], linestyle='--', color='red',
                                                           label='Início da queda' if i == 0 else "")
                                                ax.axvline(sitting_time[i], linestyle='--', color='black',
                                                           label='Início da queda' if i == 0 else "")

                                                # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                                if i + 1 < num_ciclos:
                                                    t_next_onset = time_original_kinem[onsets[i+1]]
                                                    ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                               alpha=0.3, label='Intervalo' if i == 0 else "")

                                            # Mínimos detectados
                                            for i, t in enumerate(time_original_kinem[peaks]):
                                                ax.axvline(t, linestyle='--', color='blue',
                                                           label='Mínimo' if i == 0 else "")
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Amplitude")
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, ml_acc, 'k-')

                                            largura = 0.4
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([0, 0], [0, 30], 'r--')
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Aceleração ML")
                                            ax.set_ylim([0,10])
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, ap_acc, 'k-')

                                            
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([0, 0], [0, 30], 'r--')
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Aceleração AP")
                                            ax.set_ylim([0,10])
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                            figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, v_acc, 'k-')

                                            
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            
                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([0, 0], [0, 30], 'r--',linewidth=largura)
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Aceleração V")
                                            ax.set_ylim([0,10])
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(time_interpolated_gyro,
                                                    gyro_norm_filtered, 'k-')
                                            ax.plot([0, 0], [0, 2], 'r-')
                                            # Verificação básica para evitar erros
                                            num_ciclos = min(
                                                len(onsets), len(offsets))

                                            for i in range(num_ciclos):
                                                t_onset = time_original_kinem[onsets[i]]
                                                t_offset = time_original_kinem[offsets[i]]

                                            # Linha tracejada: início
                                                ax.axvline(t_onset, linestyle='--', color='orange',
                                                           label='Início da queda' if i == 0 else "")
                                            # Linha tracejada: fim
                                                ax.axvline(t_offset, linestyle='--', color='green',
                                                           label='Fim da queda' if i == 0 else "")
                                            # Faixa entre onset e offset
                                                ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                           label='Fase de queda' if i == 0 else "")
                                                ax.axvline(standing_time[i], linestyle='--', color='red',
                                                           label='Início da queda' if i == 0 else "")
                                                ax.axvline(sitting_time[i], linestyle='--', color='black',
                                                           label='Início da queda' if i == 0 else "")

                                            # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                                if i + 1 < num_ciclos:
                                                    t_next_onset = time_original_kinem[onsets[i+1]]
                                                    ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                               alpha=0.3, label='Intervalo' if i == 0 else "")

                                                # Mínimos detectados
                                            for i, t in enumerate(time_original_kinem[peaks]):
                                                ax.axvline(t, linestyle='--', color='blue',
                                                           label='Mínimo' if i == 0 else "")
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Amplitude")
                                            st.pyplot(fig)

                                            for index,valor in enumerate(time_interpolated_gyro):
                                                if valor > time_original_kinem[onsets[0]]:
                                                    start_gyro = index-1
                                                    break
                                                    
                                            for index,valor in enumerate(ml_gyro[start_gyro:-1]):
                                                if valor > 0.15:
                                                    onset_gyro = index
                                                    break

                                            for index,valor in enumerate(time_interpolated_gyro):
                                                if valor > time_original_kinem[onsets[1]]:
                                                    start_gyro2 = index-1
                                                    break
                                                    
                                            for index in range(len(ml_gyro[0:start_gyro2]) - 1, start_gyro + 100, -1):
                                                valor = ml_gyro[index]
                                                if valor > 0.15:
                                                    offset_gyro = start_gyro2-index-1
                                                    break
                                                    
                                            for index,valor in enumerate(time_interpolated_gyro):
                                                if valor > time_original_kinem[onsets[2]]:
                                                    start_gyro3 = index-1
                                                    break

                                            for index in range(len(ml_gyro[0:start_gyro3]) - 1, start_gyro2 + 100, -1):
                                                valor = ml_gyro[index]
                                                if valor > 0.15:
                                                    offset_gyro2 = start_gyro3-index-1
                                                    break

                                            for index,valor in enumerate(time_interpolated_gyro):
                                                if valor > time_original_kinem[onsets[3]]:
                                                    start_gyro4 = index-1
                                                    break

                                            for index in range(len(ml_gyro[0:start_gyro4]) - 1, start_gyro3 + 100, -1):
                                                valor = ml_gyro[index]
                                                if valor > 0.15:
                                                    offset_gyro3 = start_gyro4-index-1
                                                    break

                                            for index in range(len(ml_gyro) - 1, start_gyro4 + 100, -1):
                                                valor = ml_gyro[index]
                                                if valor > 0.15:
                                                    offset_gyro4 = start_gyro4-index-1
                                                    break
                                                
                                            fig, ax = plt.subplots(
                                            figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated_gyro, ml_gyro, 'k-')
                                            ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro],time_interpolated_gyro[onset_gyro+start_gyro]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[start_gyro2-offset_gyro],time_interpolated_gyro[start_gyro2-offset_gyro]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro2],time_interpolated_gyro[onset_gyro+start_gyro2]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[start_gyro3-offset_gyro2],time_interpolated_gyro[start_gyro3-offset_gyro2]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro3],time_interpolated_gyro[onset_gyro+start_gyro3]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro3],time_interpolated_gyro[start_gyro4-offset_gyro3]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro4],time_interpolated_gyro[onset_gyro+start_gyro4]], [0,30], 'y--')
                                            ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro4],time_interpolated_gyro[start_gyro4-offset_gyro4]], [0,30], 'y--')
                                            
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)
                                            
                                            ax.plot([0, 0], [0, 30], 'r--')
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Velocidade angular ML")
                                            ax.set_ylim([0,3])
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated_gyro, ap_gyro, 'k-')
                                            
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)

                                            ax.plot([0, 0], [0, 30], 'r--')
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Velocidade angular AP")
                                            ax.set_ylim([0,3])
                                            st.pyplot(fig)

                                            y = v_gyro[onset_gyro+start_gyro:start_gyro2-offset_gyro]
                                            indices, propriedades = find_peaks(y)
                                            indices = indices + start_gyro + onset_gyro
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos = v_gyro[indices][picos_ordenados]
                                            momentos_picos = time_interpolated_gyro[indices][picos_ordenados]

                                            y2 = v_gyro[start_gyro2:start_gyro3-offset_gyro2]
                                            indices, propriedades = find_peaks(y2)
                                            indices = indices + start_gyro2
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos2 = v_gyro[indices][picos_ordenados]
                                            momentos_picos2 = time_interpolated_gyro[indices][picos_ordenados]

                                            y3 = v_gyro[start_gyro3:start_gyro4-offset_gyro3]
                                            indices, propriedades = find_peaks(y3)
                                            indices = indices + start_gyro3
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos3 = v_gyro[indices][picos_ordenados]
                                            momentos_picos3 = time_interpolated_gyro[indices][picos_ordenados]

                                            y4 = v_gyro[start_gyro4:offset_gyro4]
                                            indices, propriedades = find_peaks(y4)
                                            indices = indices + start_gyro4
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos4 = v_gyro[indices][picos_ordenados]
                                            momentos_picos4 = time_interpolated_gyro[indices][picos_ordenados]
                                            
                                            fig, ax = plt.subplots(
                                            figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated_gyro, v_gyro, 'k-')
                                            
                                            ax.plot(
                                                [momentos_picos[0],momentos_picos[0]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos[1],momentos_picos[1]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos2[0],momentos_picos2[0]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos2[1],momentos_picos2[1]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos3[0],momentos_picos3[0]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos3[1],momentos_picos3[1]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos4[0],momentos_picos4[0]], [0,30], 'c--')
                                            ax.plot(
                                                [momentos_picos4[1],momentos_picos4[1]], [0,30], 'c--')
                                            
                                            ax.plot(
                                                [time_original_kinem[onsets[0]],time_original_kinem[onsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[0]],time_original_kinem[offsets[0]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[1]],time_original_kinem[onsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[1]],time_original_kinem[offsets[1]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[2]],time_original_kinem[onsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[2]],time_original_kinem[offsets[2]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[onsets[3]],time_original_kinem[onsets[3]]], [0,30], 'b-',linewidth=largura)
                                            ax.plot(
                                                [time_original_kinem[offsets[3]],time_original_kinem[offsets[3]]], [0,30], 'b-',linewidth=largura)

                                            ax.plot([standing_time[0],standing_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[1],standing_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[2],standing_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([standing_time[3],standing_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([sitting_time[0],sitting_time[0]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[1],sitting_time[1]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[2],sitting_time[2]], [0,30], 'r-',linewidth=largura)
                                            ax.plot([sitting_time[3],sitting_time[3]], [0,30], 'r-',linewidth=largura)

                                            ax.plot([time_original_kinem[peaks[0]],time_original_kinem[peaks[0]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[1]],time_original_kinem[peaks[1]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[2]],time_original_kinem[peaks[2]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([time_original_kinem[peaks[3]],time_original_kinem[peaks[3]]], [0,30], 'k-',linewidth=largura)
                                            ax.plot([0, 0], [0, 5], 'r--')
                                            ax.set_xlabel("Tempo (s)")
                                            ax.set_ylabel("Velocidade angular V")
                                            ax.set_ylim([0,5])
                                            st.pyplot(fig)
                                            
                                            with col1:
                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_y, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro],time_interpolated_gyro[onset_gyro+start_gyro]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro2-offset_gyro],time_interpolated_gyro[start_gyro2-offset_gyro]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro2],time_interpolated_gyro[onset_gyro+start_gyro2]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro3-offset_gyro2],time_interpolated_gyro[start_gyro3-offset_gyro2]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro3],time_interpolated_gyro[onset_gyro+start_gyro3]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro3],time_interpolated_gyro[start_gyro4-offset_gyro3]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro4],time_interpolated_gyro[onset_gyro+start_gyro4]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro4],time_interpolated_gyro[start_gyro4-offset_gyro4]], [0,30], 'y--')
                                            
                                                ax.plot(
                                                [momentos_picos[0],momentos_picos[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos[1],momentos_picos[1]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos2[0],momentos_picos2[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos2[1],momentos_picos2[1]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos3[0],momentos_picos3[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos3[1],momentos_picos3[1]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos4[0],momentos_picos4[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos4[1],momentos_picos4[1]], [0,30], 'c--')
                                                ax.set_ylim([0,5])
                                                st.pyplot(fig)
                                                st.text(
                                                    f'Número de ciclos = {num_ciclos}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Inicio do ciclo {idx+1} = {time_original_kinem[onsets[idx]]}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Final do ciclo {idx+1} = {time_original_kinem[offsets[idx]]}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Tempo de retorno do ciclo {idx+1} = {time_original_kinem[peaks[idx]]}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração do ciclo {idx+1} = {time_original_kinem[offsets[idx]]-time_original_kinem[onsets[idx]]}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da ida do ciclo {idx+1} = {time_original_kinem[peaks[idx]] - time_original_kinem[onsets[idx]]}')
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da volta do ciclo {idx+1} = {time_original_kinem[offsets[idx]] - time_original_kinem[peaks[idx]]}')
                                            with col2:
                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Momento final da subida do ciclo {idx+1} = {standing_time[idx]}')

                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Momento inicial da descida do ciclo {idx+1} = {sitting_time[idx]}')

                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da subida do ciclo {idx+1} = {standing_time[idx]-time_original_kinem[onsets[idx]]}')

                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da descida do ciclo {idx+1} = {time_original_kinem[offsets[idx]] - sitting_time[idx]}')

                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da ida {idx+1} = {time_original_kinem[peaks[idx]] - standing_time[idx]}')

                                                for idx in np.arange(4):
                                                    st.text(
                                                        f'Duração da volta {idx+1} = {sitting_time[idx] - time_original_kinem[peaks[idx]]}')







