import streamlit as st
import scipy
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

def _safe_name(up):
    try:
        return up.name
    except Exception:
        return ""

def _safe_val(x):
    try:
        return float(x)
    except Exception:
        return x  # deixa string/None como está


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
                        cutoff = 2.5  # Frequência de corte
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
                                    fig, ax1 = plt.subplots(figsize=(10, 4))
                                    ax1.plot(time_original_kinem, disp_y, 'k-')
                                    # Verificação básica para evitar erros
                                    num_ciclos = min(len(onsets), len(offsets))
                                    for i in range(num_ciclos):
                                        t_onset = time_original_kinem[onsets[i]]
                                        t_offset = time_original_kinem[offsets[i]]
                                    # Linha tracejada: início
                                        ax1.axvline(t_onset, linestyle='--', color='orange',
                                                   label='Início da queda' if i == 0 else "")
                                    # Linha tracejada: fim
                                        ax1.axvline(t_offset, linestyle='--', color='green',
                                                   label='Fim da queda' if i == 0 else "")
                                    # Faixa entre onset e offset
                                        ax1.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                   label='Fase de queda' if i == 0 else "")

                                    # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                        #if i + 1 < num_ciclos:
                                            #t_next_onset = time_original_kinem[onsets[i+1]]
                                            #ax1.axvspan(t_offset, t_next_onset, color='lightblue',
                                                       #alpha=0.3, label='Intervalo' if i == 0 else "")

                                    # Mínimos detectados
                                    for i, t in enumerate(time_original_kinem[peaks]):
                                        ax1.axvline(t, linestyle='--', color='blue',
                                                   label='Mínimo' if i == 0 else "")

                                    ax1.set_xlabel("Tempo (s)")
                                    ax1.set_ylabel("Amplitude")
                                    ax1.set_xlim([5,25])
                                    st.pyplot(fig)
                                    with col2:
                                        fig1, ax2 = plt.subplots(figsize=(10, 4))
                                        ax2.plot(
                                            time_original_kinem, disp_z, 'k-')
                                        ax2.plot([0, 0], [0, 2], 'r-')

                                        # Verificação básica para evitar erros

                                        num_ciclos = min(
                                            len(onsets), len(offsets))

                                        for i in range(num_ciclos):
                                            t_onset = time_original_kinem[onsets[i]]
                                            t_offset = time_original_kinem[offsets[i]]

                                            #    # Linha tracejada: início
                                            ax2.axvline(t_onset, linestyle='--', color='orange',
                                                       label='Início da queda' if i == 0 else "")
                                            # Linha tracejada: fim
                                            ax2.axvline(t_offset, linestyle='--', color='green',
                                                       label='Fim da queda' if i == 0 else "")
                                            # Faixa entre onset e offset
                                            ax2.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                       label='Fase de queda' if i == 0 else "")
                                            #    # Linha tracejada: início
                                            ax2.axvline(standing_time[i], linestyle='--', color='red',
                                                       label='Início da queda' if i == 0 else "")
                                            ax2.axvline(sitting_time[i], linestyle='--', color='black',
                                                       label='Início da queda' if i == 0 else "")

                                            # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                            #if i + 1 < num_ciclos:
                                                #t_next_onset = time_original_kinem[onsets[i+1]]
                                                #ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                           #alpha=0.3, label='Intervalo' if i == 0 else "")

                                        # Mínimos detectados
                                        for i, t in enumerate(time_original_kinem[peaks]):
                                            ax2.axvline(t, linestyle='--', color='blue',
                                                       label='Mínimo' if i == 0 else "")

                                        ax2.set_xlabel("Tempo (s)")
                                        ax2.set_ylabel("Amplitude")
                                        ax2.set_ylim([0.5,1.5])
                                        ax2.set_xlim([5,25])
                                        st.pyplot(fig1)
                                        with col3:
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

                                            for index in range(len(ml_gyro) - 600, start_gyro4 + 100, -1):
                                                valor = ml_gyro[index]
                                                if valor > 0.15:
                                                    offset_gyro4 = start_gyro4-index-1
                                                    break
                                            
                                            fig3, ax3 = plt.subplots(
                                                figsize=(10, 4))
                                            ax3.plot(
                                                time_interpolated, acc_norm_filtered, 'k-')
                                            ax3.plot([0, 0], [0, 30], 'r--')
                                            # Verificação básica para evitar erros
                                            num_ciclos = min(
                                                len(onsets), len(offsets))

                                            for i in range(num_ciclos):
                                                t_onset = time_original_kinem[onsets[i]]
                                                t_offset = time_original_kinem[offsets[i]]

                                                # Linha tracejada: início
                                                ax3.axvline(t_onset, linestyle='--', color='orange',
                                                           label='Início da queda' if i == 0 else "")
                                                # Linha tracejada: fim
                                                ax3.axvline(t_offset, linestyle='--', color='green',
                                                           label='Fim da queda' if i == 0 else "")
                                                # Faixa entre onset e offset
                                                ax3.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                                                           label='Fase de queda' if i == 0 else "")

                                                ax3.axvline(standing_time[i], linestyle='--', color='red',
                                                           label='Início da queda' if i == 0 else "")
                                                ax3.axvline(sitting_time[i], linestyle='--', color='black',
                                                           label='Início da queda' if i == 0 else "")

                                                # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                                                #if i + 1 < num_ciclos:
                                                   # t_next_onset = time_original_kinem[onsets[i+1]]
                                                   # ax.axvspan(t_offset, t_next_onset, color='lightblue',
                                                              # alpha=0.3, label='Intervalo' if i == 0 else "")

                                            # Mínimos detectados
                                            for i, t in enumerate(time_original_kinem[peaks]):
                                                ax.axvline(t, linestyle='--', color='blue',
                                                           label='Mínimo' if i == 0 else "")
                                            ax3.set_xlabel("Tempo (s)")
                                            ax3.set_ylabel("Amplitude")
                                            ax3.set_ylim([0,10])
                                            ax3.set_xlim([5,25])
                                            st.pyplot(fig3)

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

                                            y = ap_acc[onset_gyro+start_gyro:start_gyro2-offset_gyro]
                                            indices, propriedades = find_peaks(y)
                                            indices = indices + start_gyro + onset_gyro
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ap_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picosap_acc = ap_acc[indices][picos_ordenados]
                                            momentos_picosap_acc = time_interpolated[indices][picos_ordenados]
                                            if momentos_picosap_acc[0] > momentos_picosap_acc[1]:
                                                t_acc_ap_pico_levantar_ciclo_1 = momentos_picosap_acc[1]
                                                t_acc_ap_pico_sentar_ciclo_1 = momentos_picosap_acc[0]
                                            else:
                                                t_acc_ap_pico_sentar_ciclo_1 = momentos_picosap_acc[1]
                                                t_acc_ap_pico_levantar_ciclo_1 = momentos_picosap_acc[0]
                                                
                                            y2 = ap_acc[start_gyro2:start_gyro3-offset_gyro2]
                                            indices, propriedades = find_peaks(y2)
                                            indices = indices + start_gyro2
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ap_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos2ap_acc = ap_acc[indices][picos_ordenados]
                                            momentos_picos2ap_acc = time_interpolated[indices][picos_ordenados]
                                            if momentos_picos2ap_acc[0] > momentos_picos2ap_acc[1]:
                                                t_acc_ap_pico_levantar_ciclo_2 = momentos_picos2ap_acc[1]
                                                t_acc_ap_pico_sentar_ciclo_2 = momentos_picos2ap_acc[0]
                                            else:
                                                t_acc_ap_pico_sentar_ciclo_2 = momentos_picos2ap_acc[1]
                                                t_acc_ap_pico_levantar_ciclo_2 = momentos_picos2ap_acc[0]

                                            y3 = ap_acc[start_gyro3:start_gyro4-offset_gyro3]
                                            indices, propriedades = find_peaks(y3)
                                            indices = indices + start_gyro3
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ap_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos3ap_acc = ap_acc[indices][picos_ordenados]
                                            momentos_picos3ap_acc = time_interpolated[indices][picos_ordenados]
                                            if momentos_picos3ap_acc[0] > momentos_picos3ap_acc[1]:
                                                t_acc_ap_pico_levantar_ciclo_3 = momentos_picos3ap_acc[1]
                                                t_acc_ap_pico_sentar_ciclo_3 = momentos_picos3ap_acc[0]
                                            else:
                                                t_acc_ap_pico_sentar_ciclo_3 = momentos_picos3ap_acc[1]
                                                t_acc_ap_pico_levantar_ciclo_3 = momentos_picos3ap_acc[0]

                                            y4 = ap_acc[start_gyro4:offset_gyro4]
                                            indices, propriedades = find_peaks(y4)
                                            indices = indices + start_gyro4
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ap_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos4ap_acc = ap_acc[indices][picos_ordenados]
                                            momentos_picos4ap_acc = time_interpolated[indices][picos_ordenados]
                                            if momentos_picos4ap_acc[0] > momentos_picos4ap_acc[1]:
                                                t_acc_ap_pico_levantar_ciclo_4 = momentos_picos4ap_acc[1]
                                                t_acc_ap_pico_sentar_ciclo_4 = momentos_picos4ap_acc[0]
                                            else:
                                                t_acc_ap_pico_sentar_ciclo_4 = momentos_picos4ap_acc[1]
                                                t_acc_ap_pico_levantar_ciclo_4 = momentos_picos4ap_acc[0]

                                            y = v_acc[onset_gyro+start_gyro:start_gyro2-offset_gyro]
                                            indices, propriedades = find_peaks(y)
                                            indices = indices + start_gyro + onset_gyro
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picosv_acc = v_acc[indices][picos_ordenados]
                                            momentos_picosv_acc = time_interpolated[indices][picos_ordenados]

                                            if momentos_picosv_acc[0] > momentos_picosv_acc[1]:
                                                t_acc_v_pico_levantar_ciclo_1 = momentos_picosv_acc[1]
                                                t_acc_v_pico_sentar_ciclo_1 = momentos_picosv_acc[0]
                                            else:
                                                t_acc_v_pico_sentar_ciclo_1 = momentos_picosv_acc[1]
                                                t_acc_v_pico_levantar_ciclo_1 = momentos_picosv_acc[0]

                                            y2 = v_acc[start_gyro2:start_gyro3-offset_gyro2]
                                            indices, propriedades = find_peaks(y2)
                                            indices = indices + start_gyro2
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos2v_acc = v_acc[indices][picos_ordenados]
                                            momentos_picos2v_acc = time_interpolated[indices][picos_ordenados]

                                            if momentos_picos2v_acc[0] > momentos_picos2v_acc[1]:
                                                t_acc_v_pico_levantar_ciclo_2 = momentos_picos2v_acc[1]
                                                t_acc_v_pico_sentar_ciclo_2 = momentos_picos2v_acc[0]
                                            else:
                                                t_acc_v_pico_sentar_ciclo_2 = momentos_picos2v_acc[1]
                                                t_acc_v_pico_levantar_ciclo_2 = momentos_picos2v_acc[0]

                                            y3 = v_acc[start_gyro3:start_gyro4-offset_gyro3]
                                            indices, propriedades = find_peaks(y3)
                                            indices = indices + start_gyro3
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos3v_acc = v_acc[indices][picos_ordenados]
                                            momentos_picos3v_acc = time_interpolated[indices][picos_ordenados]

                                            if momentos_picos3v_acc[0] > momentos_picos3v_acc[1]:
                                                t_acc_v_pico_levantar_ciclo_3 = momentos_picos3v_acc[1]
                                                t_acc_v_pico_sentar_ciclo_3 = momentos_picos3v_acc[0]
                                            else:
                                                t_acc_v_pico_sentar_ciclo_3 = momentos_picos3v_acc[1]
                                                t_acc_v_pico_levantar_ciclo_3 = momentos_picos3v_acc[0]
                                                
                                            y4 = v_acc[start_gyro4:offset_gyro4]
                                            indices, propriedades = find_peaks(y4)
                                            indices = indices + start_gyro4
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_acc[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos4v_acc = v_acc[indices][picos_ordenados]
                                            momentos_picos4v_acc = time_interpolated[indices][picos_ordenados]

                                            if momentos_picos4v_acc[0] > momentos_picos4v_acc[1]:
                                                t_acc_v_pico_levantar_ciclo_4 = momentos_picos4v_acc[1]
                                                t_acc_v_pico_sentar_ciclo_4 = momentos_picos4v_acc[0]
                                            else:
                                                t_acc_v_pico_sentar_ciclo_4 = momentos_picos2v_acc[1]
                                                t_acc_v_pico_levantar_ciclo_4 = momentos_picos2v_acc[0]
                                            
                                            fig, ax = plt.subplots(
                                                figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, ap_acc, 'k-')

                                            ax.plot(
                                                [momentos_picosap_acc[0],momentos_picosap_acc[0]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picosap_acc[1],momentos_picosap_acc[1]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos2ap_acc[0],momentos_picos2ap_acc[0]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos2ap_acc[1],momentos_picos2ap_acc[1]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos3ap_acc[0],momentos_picos3ap_acc[0]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos3ap_acc[1],momentos_picos3ap_acc[1]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos4ap_acc[0],momentos_picos4ap_acc[0]], [0,30], 'b--')
                                            ax.plot(
                                                [momentos_picos4ap_acc[1],momentos_picos4ap_acc[1]], [0,30], 'b--')
                                            
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
                                            ax.set_xlim([5,25])
                                            st.pyplot(fig)

                                            fig, ax = plt.subplots(
                                            figsize=(10, 4))
                                            ax.plot(
                                                time_interpolated, v_acc, 'k-')

                                            ax.plot(
                                                [momentos_picosv_acc[0],momentos_picosv_acc[0]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picosv_acc[1],momentos_picosv_acc[1]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos2v_acc[0],momentos_picos2v_acc[0]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos2v_acc[1],momentos_picos2v_acc[1]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos3v_acc[0],momentos_picos3v_acc[0]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos3v_acc[1],momentos_picos3v_acc[1]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos4v_acc[0],momentos_picos4v_acc[0]], [0,30], 'k--')
                                            ax.plot(
                                                [momentos_picos4v_acc[1],momentos_picos4v_acc[1]], [0,30], 'k--')

                                            
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
                                            ax.set_xlim([5,25])
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
                                            ax.set_xlim([5,25])
                                            st.pyplot(fig)


                                            y = ml_gyro[onset_gyro+start_gyro:start_gyro2-offset_gyro]
                                            indices, propriedades = find_peaks(y)
                                            indices = indices + start_gyro + onset_gyro
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ml_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picosml = ml_gyro[indices][picos_ordenados]
                                            momentos_picosml = time_interpolated_gyro[indices][picos_ordenados]
                                            
                                            if momentos_picosml[0] > momentos_picosml[1]:
                                                t_gyro_ml_pico_levantar_ciclo_1 = momentos_picosml[1]
                                                t_gyro_ml_pico_sentar_ciclo_1 = momentos_picosml[0]
                                            else:
                                                t_gyro_ml_pico_sentar_ciclo_1 = momentos_picosml[1]
                                                t_gyro_ml_pico_levantar_ciclo_1 = momentos_picosml[0]

                                            y2 = ml_gyro[start_gyro2:start_gyro3-offset_gyro2]
                                            indices, propriedades = find_peaks(y2)
                                            indices = indices + start_gyro2
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ml_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos2ml = ml_gyro[indices][picos_ordenados]
                                            momentos_picos2ml = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos2ml[0] > momentos_picos2ml[1]:
                                                t_gyro_ml_pico_levantar_ciclo_2 = momentos_picos2ml[1]
                                                t_gyro_ml_pico_sentar_ciclo_2 = momentos_picos2ml[0]
                                            else:
                                                t_gyro_ml_pico_sentar_ciclo_2 = momentos_picos2ml[1]
                                                t_gyro_ml_pico_levantar_ciclo_2 = momentos_picos2ml[0]

                                            y3 = ml_gyro[start_gyro3:start_gyro4-offset_gyro3]
                                            indices, propriedades = find_peaks(y3)
                                            indices = indices + start_gyro3
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ml_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos3ml = ml_gyro[indices][picos_ordenados]
                                            momentos_picos3ml = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos3ml[0] > momentos_picos3ml[1]:
                                                t_gyro_ml_pico_levantar_ciclo_3 = momentos_picos3ml[1]
                                                t_gyro_ml_pico_sentar_ciclo_3 = momentos_picos3ml[0]
                                            else:
                                                t_gyro_ml_pico_sentar_ciclo_3 = momentos_picos3ml[1]
                                                t_gyro_ml_pico_levantar_ciclo_3 = momentos_picos3ml[0]


                                            y4 = ml_gyro[start_gyro4:offset_gyro4]
                                            indices, propriedades = find_peaks(y4)
                                            indices = indices + start_gyro4
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(ml_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos4ml = ml_gyro[indices][picos_ordenados]
                                            momentos_picos4ml = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos4ml[0] > momentos_picos4ml[1]:
                                                t_gyro_ml_pico_levantar_ciclo_4 = momentos_picos4ml[1]
                                                t_gyro_ml_pico_sentar_ciclo_4 = momentos_picos4ml[0]
                                            else:
                                                t_gyro_ml_pico_sentar_ciclo_4 = momentos_picos4ml[1]
                                                t_gyro_ml_pico_levantar_ciclo_4 = momentos_picos4ml[0]

                                                
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
                                                [momentos_picosml[0],momentos_picosml[0]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picosml[1],momentos_picosml[1]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos2ml[0],momentos_picos2ml[0]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos2ml[1],momentos_picos2ml[1]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos3ml[0],momentos_picos3ml[0]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos3ml[1],momentos_picos3ml[1]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos4ml[0],momentos_picos4ml[0]], [0,30], 'r--')
                                            ax.plot(
                                                [momentos_picos4ml[1],momentos_picos4ml[1]], [0,30], 'r--')
                                            
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
                                            ax.set_xlim([5,25])
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
                                            ax.set_xlim([5,25])
                                            st.pyplot(fig)

                                            y = v_gyro[onset_gyro+start_gyro:start_gyro2-offset_gyro]
                                            indices, propriedades = find_peaks(y)
                                            indices = indices + start_gyro + onset_gyro
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos = v_gyro[indices][picos_ordenados]
                                            momentos_picos = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos[0] > momentos_picos[1]:
                                                t_gyro_v_pico_giro3m_ciclo_1 = momentos_picos[1]
                                                t_gyro_v_pico_giro6m_ciclo_1 = momentos_picos[0]
                                            else:
                                                t_gyro_v_pico_giro6m_ciclo_1 = momentos_picos[1]
                                                t_gyro_v_pico_giro3m_ciclo_1 = momentos_picos[0]


                                            y2 = v_gyro[start_gyro2:start_gyro3-offset_gyro2]
                                            indices, propriedades = find_peaks(y2)
                                            indices = indices + start_gyro2
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos2 = v_gyro[indices][picos_ordenados]
                                            momentos_picos2 = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos2[0] > momentos_picos2[1]:
                                                t_gyro_v_pico_giro3m_ciclo_2 = momentos_picos2[1]
                                                t_gyro_v_pico_giro6m_ciclo_2 = momentos_picos2[0]
                                            else:
                                                t_gyro_v_pico_giro6m_ciclo_2 = momentos_picos2[1]
                                                t_gyro_v_pico_giro3m_ciclo_2 = momentos_picos2[0]

                                            y3 = v_gyro[start_gyro3:start_gyro4-offset_gyro3]
                                            indices, propriedades = find_peaks(y3)
                                            indices = indices + start_gyro3
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos3 = v_gyro[indices][picos_ordenados]
                                            momentos_picos3 = time_interpolated_gyro[indices][picos_ordenados]

                                            if momentos_picos3[0] > momentos_picos3[1]:
                                                t_gyro_v_pico_giro3m_ciclo_3 = momentos_picos3[1]
                                                t_gyro_v_pico_giro6m_ciclo_3 = momentos_picos3[0]
                                            else:
                                                t_gyro_v_pico_giro6m_ciclo_3 = momentos_picos3[1]
                                                t_gyro_v_pico_giro3m_ciclo_3 = momentos_picos3[0]

                                            y4 = v_gyro[start_gyro4:offset_gyro4]
                                            indices, propriedades = find_peaks(y4)
                                            indices = indices + start_gyro4
                                            # Ordenar picos por altura
                                            picos_ordenados = np.argsort(v_gyro[indices])[-2:]  # dois maiores picos
                                            # Extrair valores e momentos
                                            maiores_picos4 = v_gyro[indices][picos_ordenados]
                                            momentos_picos4 = time_interpolated_gyro[indices][picos_ordenados]
                                            
                                            if momentos_picos4[0] > momentos_picos4[1]:
                                                t_gyro_v_pico_giro3m_ciclo_4 = momentos_picos4[1]
                                                t_gyro_v_pico_giro6m_ciclo_4 = momentos_picos4[0]
                                            else:
                                                t_gyro_v_pico_giro6m_ciclo_4 = momentos_picos4[1]
                                                t_gyro_v_pico_giro3m_ciclo_4 = momentos_picos4[0]
                                            
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
                                            ax.set_xlim([5,25])
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

                                                ax.plot(
                                                [momentos_picosml[0],momentos_picosml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picosml[1],momentos_picosml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[0],momentos_picos2ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[1],momentos_picos2ml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[0],momentos_picos3ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[1],momentos_picos3ml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[0],momentos_picos4ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[1],momentos_picos4ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picosap_acc[0],momentos_picosap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picosap_acc[1],momentos_picosap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[0],momentos_picos2ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[1],momentos_picos2ap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[0],momentos_picos3ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[1],momentos_picos3ap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[0],momentos_picos4ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[1],momentos_picos4ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picosv_acc[0],momentos_picosv_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picosv_acc[1],momentos_picosv_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[0],momentos_picos2v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[1],momentos_picos2v_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[0],momentos_picos3v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[1],momentos_picos3v_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[0],momentos_picos4v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[1],momentos_picos4v_acc[1]], [0,30], 'k--')
                                            
                                                ax.set_ylim([-0.5,5])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_y, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro],time_interpolated_gyro[onset_gyro+start_gyro]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro2-offset_gyro],time_interpolated_gyro[start_gyro2-offset_gyro]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos[0],momentos_picos[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos[1],momentos_picos[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picosml[0],momentos_picosml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picosml[1],momentos_picosml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picosap_acc[0],momentos_picosap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picosap_acc[1],momentos_picosap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picosv_acc[0],momentos_picosv_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picosv_acc[1],momentos_picosv_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro-100],time_interpolated_gyro[start_gyro2-offset_gyro+100]])
                                                ax.set_ylim([-0.5,5])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_y, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro2],time_interpolated_gyro[onset_gyro+start_gyro2]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro3-offset_gyro2],time_interpolated_gyro[start_gyro3-offset_gyro2]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos2[0],momentos_picos2[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos2[1],momentos_picos2[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos2ml[0],momentos_picos2ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[1],momentos_picos2ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos2ap_acc[0],momentos_picos2ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[1],momentos_picos2ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos2v_acc[0],momentos_picos2v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[1],momentos_picos2v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro2-100],time_interpolated_gyro[start_gyro3-offset_gyro2+100]])
                                                ax.set_ylim([-0.5,5])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_y, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro3],time_interpolated_gyro[onset_gyro+start_gyro3]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro3],time_interpolated_gyro[start_gyro4-offset_gyro3]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos3[0],momentos_picos3[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos3[1],momentos_picos3[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos3ml[0],momentos_picos3ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[1],momentos_picos3ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos3ap_acc[0],momentos_picos3ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[1],momentos_picos3ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos3v_acc[0],momentos_picos3v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[1],momentos_picos3v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro3-100],time_interpolated_gyro[start_gyro4-offset_gyro3+100]])
                                                ax.set_ylim([-0.5,5])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_y, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro4],time_interpolated_gyro[onset_gyro+start_gyro4]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[offset_gyro4],time_interpolated_gyro[offset_gyro4]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos4[0],momentos_picos4[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos4[1],momentos_picos4[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos4ml[0],momentos_picos4ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[1],momentos_picos4ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos4ap_acc[0],momentos_picos4ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[1],momentos_picos4ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos4v_acc[0],momentos_picos4v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[1],momentos_picos4v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro4-100],time_interpolated_gyro[offset_gyro4+100]])
                                                ax.set_ylim([-0.5,5])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)
                                                
                                            with col2:
                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(
                                                time_original_kinem, disp_z, 'k-')
                                                ax.plot([0, 0], [0, 2], 'r-')

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

                                                ax.plot(
                                                [momentos_picosml[0],momentos_picosml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picosml[1],momentos_picosml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[0],momentos_picos2ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[1],momentos_picos2ml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[0],momentos_picos3ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[1],momentos_picos3ml[1]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[0],momentos_picos4ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[1],momentos_picos4ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picosap_acc[0],momentos_picosap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picosap_acc[1],momentos_picosap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[0],momentos_picos2ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[1],momentos_picos2ap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[0],momentos_picos3ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[1],momentos_picos3ap_acc[1]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[0],momentos_picos4ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[1],momentos_picos4ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picosv_acc[0],momentos_picosv_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picosv_acc[1],momentos_picosv_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[0],momentos_picos2v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[1],momentos_picos2v_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[0],momentos_picos3v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[1],momentos_picos3v_acc[1]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[0],momentos_picos4v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[1],momentos_picos4v_acc[1]], [0,30], 'k--')
                                                ax.set_ylim([0,2])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_z, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro],time_interpolated_gyro[onset_gyro+start_gyro]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro2-offset_gyro],time_interpolated_gyro[start_gyro2-offset_gyro]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos[0],momentos_picos[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos[1],momentos_picos[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picosml[0],momentos_picosml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picosml[1],momentos_picosml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picosap_acc[0],momentos_picosap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picosap_acc[1],momentos_picosap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picosv_acc[0],momentos_picosv_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picosv_acc[1],momentos_picosv_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro-100],time_interpolated_gyro[start_gyro2-offset_gyro+100]])
                                                ax.set_ylim([0.6,1.2])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_z, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro2],time_interpolated_gyro[onset_gyro+start_gyro2]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro3-offset_gyro2],time_interpolated_gyro[start_gyro3-offset_gyro2]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos2[0],momentos_picos2[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos2[1],momentos_picos2[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos2ml[0],momentos_picos2ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos2ml[1],momentos_picos2ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos2ap_acc[0],momentos_picos2ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos2ap_acc[1],momentos_picos2ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos2v_acc[0],momentos_picos2v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos2v_acc[1],momentos_picos2v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro2-100],time_interpolated_gyro[start_gyro3-offset_gyro2+100]])
                                                ax.set_ylim([0.6,1.2])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_z, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro3],time_interpolated_gyro[onset_gyro+start_gyro3]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[start_gyro4-offset_gyro3],time_interpolated_gyro[start_gyro4-offset_gyro3]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos3[0],momentos_picos3[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos3[1],momentos_picos3[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos3ml[0],momentos_picos3ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos3ml[1],momentos_picos3ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos3ap_acc[0],momentos_picos3ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos3ap_acc[1],momentos_picos3ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos3v_acc[0],momentos_picos3v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos3v_acc[1],momentos_picos3v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro3-100],time_interpolated_gyro[start_gyro4-offset_gyro3+100]])
                                                ax.set_ylim([0.6,1.2])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)

                                                fig, ax = plt.subplots(figsize=(10, 4))
                                                ax.plot(time_original_kinem, disp_z, 'k-')

                                                ax.plot(
                                                [time_interpolated_gyro[onset_gyro+start_gyro4],time_interpolated_gyro[onset_gyro+start_gyro4]], [0,30], 'y--')
                                                ax.plot(
                                                [time_interpolated_gyro[offset_gyro4],time_interpolated_gyro[offset_gyro4]], [0,30], 'y--')
                                                                                            
                                                ax.plot(
                                                [momentos_picos4[0],momentos_picos4[0]], [0,30], 'c--')
                                                ax.plot(
                                                [momentos_picos4[1],momentos_picos4[1]], [0,30], 'c--')

                                                ax.plot(
                                                [momentos_picos4ml[0],momentos_picos4ml[0]], [0,30], 'r--')
                                                ax.plot(
                                                [momentos_picos4ml[1],momentos_picos4ml[1]], [0,30], 'r--')

                                                ax.plot(
                                                [momentos_picos4ap_acc[0],momentos_picos4ap_acc[0]], [0,30], 'b--')
                                                ax.plot(
                                                [momentos_picos4ap_acc[1],momentos_picos4ap_acc[1]], [0,30], 'b--')

                                                ax.plot(
                                                [momentos_picos4v_acc[0],momentos_picos4v_acc[0]], [0,30], 'k--')
                                                ax.plot(
                                                [momentos_picos4v_acc[1],momentos_picos4v_acc[1]], [0,30], 'k--')
                                                ax.set_xlim([time_interpolated_gyro[onset_gyro+start_gyro4-100],time_interpolated_gyro[offset_gyro4+100]])
                                                ax.set_ylim([0.6,1.2])
                                                ax.set_xlim([5,25])
                                                st.pyplot(fig)
                                                
                                            st.markdown("---")
                                            st.subheader("📤 Exportação das variáveis de desempenho")
                                            pares = [
                                                # Identificação dos arquivos
                                                ("arquivo_kinem", _safe_name(uploaded_file_kinem) if 'uploaded_file_kinem' in locals() else ""),
                                                ("arquivo_acc",   _safe_name(uploaded_file_acc)   if 'uploaded_file_acc'   in locals() else ""),
                                                ("arquivo_gyro",  _safe_name(uploaded_file_gyro)  if 'uploaded_file_gyro'  in locals() else ""),
                                            
                                                # Exemplos gerais
                                                ("num_ciclos", int(num_ciclos) if 'num_ciclos' in locals() else None),
                                            
                                                # Exemplos do Ciclo 1 (adicione mais ciclos duplicando e renomeando as chaves)
                                                ("Início do ciclo 1 Cinemática",   _safe_val(time_original_kinem[onsets[0]]) if 'onsets' in locals() and len(onsets) > 0 else None),
                                                ("Momento do retorno do ciclo 1 Cinemática",    _safe_val(time_original_kinem[peaks[0]])  if 'peaks'  in locals() and len(peaks)  > 0 else None),
                                                ("Final do ciclo 1 Cinemática",  _safe_val(time_original_kinem[offsets[0]]) if 'offsets' in locals() and len(offsets) > 0 else None),
                                                ("Momento final do levantar do ciclo 1 Cinemática", _safe_val(standing_time[0]) if 'standing_time' in locals() and len(standing_time) > 0 else None),
                                                ("Momento inicial do sentar do ciclo 1 Cinemática",  _safe_val(sitting_time[0])  if 'sitting_time'  in locals() and len(sitting_time)  > 0 else None),
                                                
                                                ("Início do ciclo 2 Cinemática",   _safe_val(time_original_kinem[onsets[1]]) if 'onsets' in locals() and len(onsets) > 0 else None),
                                                ("Momento do retorno do ciclo 2 Cinemática",    _safe_val(time_original_kinem[peaks[1]])  if 'peaks'  in locals() and len(peaks)  > 0 else None),
                                                ("Final do ciclo 2 Cinemática",  _safe_val(time_original_kinem[offsets[1]]) if 'offsets' in locals() and len(offsets) > 0 else None),
                                                ("Momento final do levantar do ciclo 2 Cinemática", _safe_val(standing_time[1]) if 'standing_time' in locals() and len(standing_time) > 0 else None),
                                                ("Momento inicial do sentar do ciclo 2 Cinemática",  _safe_val(sitting_time[1])  if 'sitting_time'  in locals() and len(sitting_time)  > 0 else None),

                                                ("Início do ciclo 3 Cinemática",   _safe_val(time_original_kinem[onsets[2]]) if 'onsets' in locals() and len(onsets) > 0 else None),
                                                ("Momento do retorno do ciclo 3 Cinemática",    _safe_val(time_original_kinem[peaks[2]])  if 'peaks'  in locals() and len(peaks)  > 0 else None),
                                                ("Final do ciclo 3 Cinemática",  _safe_val(time_original_kinem[offsets[2]]) if 'offsets' in locals() and len(offsets) > 0 else None),
                                                ("Momento final do levantar do ciclo 3 Cinemática", _safe_val(standing_time[2]) if 'standing_time' in locals() and len(standing_time) > 0 else None),
                                                ("Momento inicial do sentar do ciclo 3 Cinemática",  _safe_val(sitting_time[2])  if 'sitting_time'  in locals() and len(sitting_time)  > 0 else None),

                                                ("Início do ciclo 4 Cinemática",   _safe_val(time_original_kinem[onsets[3]]) if 'onsets' in locals() and len(onsets) > 0 else None),
                                                ("Momento do retorno do ciclo 4 Cinemática",    _safe_val(time_original_kinem[peaks[3]])  if 'peaks'  in locals() and len(peaks)  > 0 else None),
                                                ("Final do ciclo 4 Cinemática",  _safe_val(time_original_kinem[offsets[3]]) if 'offsets' in locals() and len(offsets) > 0 else None),
                                                ("Momento final do levantar do ciclo 4 Cinemática", _safe_val(standing_time[3]) if 'standing_time' in locals() and len(standing_time) > 0 else None),
                                                ("Momento inicial do sentar do ciclo 4 Cinemática",  _safe_val(sitting_time[3])  if 'sitting_time'  in locals() and len(sitting_time)  > 0 else None),

                                                ("Momento de pico de aceleração AP durante o levantar do ciclo 1 Acelerômetro",   _safe_val(t_acc_ap_pico_levantar_ciclo_1)),
                                                ("Momento de pico de aceleração AP durante o levantar do ciclo 2 Acelerômetro",   _safe_val(t_acc_ap_pico_levantar_ciclo_2)),
                                                ("Momento de pico de aceleração AP durante o levantar do ciclo 3 Acelerômetro",   _safe_val(t_acc_ap_pico_levantar_ciclo_3)),
                                                ("Momento de pico de aceleração AP durante o levantar do ciclo 4 Acelerômetro",   _safe_val(t_acc_ap_pico_levantar_ciclo_4)),
                                                
                                                ("Momento de pico de aceleração AP durante o sentar do ciclo 1 Acelerômetro",   _safe_val(t_acc_ap_pico_sentar_ciclo_1)),
                                                ("Momento de pico de aceleração AP durante o sentar do ciclo 2 Acelerômetro",   _safe_val(t_acc_ap_pico_sentar_ciclo_2)),
                                                ("Momento de pico de aceleração AP durante o sentar do ciclo 3 Acelerômetro",   _safe_val(t_acc_ap_pico_sentar_ciclo_3)),
                                                ("Momento de pico de aceleração AP durante o sentar do ciclo 4 Acelerômetro",   _safe_val(t_acc_ap_pico_sentar_ciclo_4)),
                                                
                                                ("Momento de pico de aceleração V durante o levantar do ciclo 1 Acelerômetro",   _safe_val(t_acc_v_pico_levantar_ciclo_1)),
                                                ("Momento de pico de aceleração V durante o levantar do ciclo 2 Acelerômetro",   _safe_val(t_acc_v_pico_levantar_ciclo_2)),
                                                ("Momento de pico de aceleração V durante o levantar do ciclo 3 Acelerômetro",   _safe_val(t_acc_v_pico_levantar_ciclo_3)),
                                                ("Momento de pico de aceleração V durante o levantar do ciclo 4 Acelerômetro",   _safe_val(t_acc_v_pico_levantar_ciclo_4)),

                                                ("Momento de pico de aceleração V durante o sentar do ciclo 1 Acelerômetro",   _safe_val(t_acc_v_pico_sentar_ciclo_1)),
                                                ("Momento de pico de aceleração V durante o sentar do ciclo 2 Acelerômetro",   _safe_val(t_acc_v_pico_sentar_ciclo_2)),
                                                ("Momento de pico de aceleração V durante o sentar do ciclo 3 Acelerômetro",   _safe_val(t_acc_v_pico_sentar_ciclo_3)),
                                                ("Momento de pico de aceleração V durante o sentar do ciclo 4 Acelerômetro",   _safe_val(t_acc_v_pico_sentar_ciclo_4)),

                                                ("Momento de pico de velocidade angular ML durante o levantar do ciclo 1 Giroscópio",   _safe_val(t_gyro_ml_pico_levantar_ciclo_1)),
                                                ("Momento de pico de velocidade angular ML durante o levantar do ciclo 2 Giroscópio",   _safe_val(t_gyro_ml_pico_levantar_ciclo_2)),
                                                ("Momento de pico de velocidade angular ML durante o levantar do ciclo 3 Giroscópio",   _safe_val(t_gyro_ml_pico_levantar_ciclo_3)),
                                                ("Momento de pico de velocidade angular ML durante o levantar do ciclo 4 Giroscópio",   _safe_val(t_gyro_ml_pico_levantar_ciclo_4)),

                                                ("Momento de pico de velocidade angular ML durante o sentar do ciclo 1 Giroscópio",   _safe_val(t_gyro_ml_pico_sentar_ciclo_1)),
                                                ("Momento de pico de velocidade angular ML durante o sentar do ciclo 2 Giroscópio",   _safe_val(t_gyro_ml_pico_sentar_ciclo_2)),
                                                ("Momento de pico de velocidade angular ML durante o sentar do ciclo 3 Giroscópio",   _safe_val(t_gyro_ml_pico_sentar_ciclo_3)),
                                                ("Momento de pico de velocidade angular ML durante o sentar do ciclo 4 Giroscópio",   _safe_val(t_gyro_ml_pico_sentar_ciclo_4)),

                                                ("Momento de pico de velocidade angular V durante o giro em 3 m do ciclo 1 Giroscópio",   _safe_val(t_gyro_v_pico_giro3m_ciclo_1)),
                                                ("Momento de pico de velocidade angular V durante o giro em 3 m do ciclo 2 Giroscópio",   _safe_val(t_gyro_v_pico_giro3m_ciclo_2)),
                                                ("Momento de pico de velocidade angular V durante o giro em 3 m do ciclo 3 Giroscópio",   _safe_val(t_gyro_v_pico_giro3m_ciclo_3)),
                                                ("Momento de pico de velocidade angular V durante o giro em 3 m do ciclo 4 Giroscópio",   _safe_val(t_gyro_v_pico_giro3m_ciclo_4)),

                                                ("Momento de pico de velocidade angular V durante o giro em 6 m do ciclo 1 Giroscópio",   _safe_val(t_gyro_v_pico_giro6m_ciclo_1)),
                                                ("Momento de pico de velocidade angular V durante o giro em 6 m do ciclo 2 Giroscópio",   _safe_val(t_gyro_v_pico_giro6m_ciclo_2)),
                                                ("Momento de pico de velocidade angular V durante o giro em 6 m do ciclo 3 Giroscópio",   _safe_val(t_gyro_v_pico_giro6m_ciclo_3)),
                                                ("Momento de pico de velocidade angular V durante o giro em 6 m do ciclo 4 Giroscópio",   _safe_val(t_gyro_v_pico_giro6m_ciclo_4)),

                                                
                                                
                                            ]
                                            
                                            # Monta DataFrame 2 colunas sem usar for
                                            df_export_2cols = pd.DataFrame(
                                                {"variavel": [p[0] for p in pares],
                                                 "valor":    [p[1] for p in pares]}
                                            )
                                            
                                            # Prévia
                                            st.caption("Prévia do arquivo (variável, valor)")
                                            st.dataframe(df_export_2cols, use_container_width=True)
                                            
                                            # Botões de download (CSV e Excel)
                                            csv_bytes = df_export_2cols.to_csv(index=False).encode("utf-8")
                                            st.download_button(
                                                label="⬇️ Baixar CSV (variável, valor)",
                                                data=csv_bytes,
                                                file_name="desempenho_2colunas.csv",
                                                mime="text/csv",
                                            )
                                            
                                            






















































