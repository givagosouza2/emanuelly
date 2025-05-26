import streamlit as st
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

        original_fs = 120
        new_fs = 100
        cutoff = 2
        time_original_kinem = np.arange(0, len(disp_y)) / original_fs

        baseline = np.mean(disp_z[100:500])
        sd_baseline = np.std(disp_z[100:500])
        for index, value in enumerate(disp_z[100:500]):
            if value > baseline + 2*sd_baseline or value < baseline + 2*sd_baseline:
                time_original_kinem = time_original_kinem - \
                    time_original_kinem[index+100]
                break
        # Diferença entre tempos consecutivos
        delta_t = np.diff(time_original_kinem)
        # Diferença entre deslocamentos consecutivos
        delta_s = np.diff(disp_x)
        velocidade_z = delta_s / delta_t
        velocidade_z = np.insert(velocidade_z, 0, 0)

        delta_s = np.diff(disp_y)
        velocidade_y = delta_s / delta_t
        velocidade_y = np.insert(velocidade_y, 0, 0)

        delta_s = np.diff(disp_z)
        velocidade_z = delta_s / delta_t
        velocidade_z = np.insert(velocidade_z, 0, 0)

        delta_v = np.diff(velocidade_y)
        aceleracao_y = delta_v / delta_t
        aceleracao_y = np.insert(aceleracao_y, 0, 0)
        aceleracao_y_filtered = low_pass_filter(
            aceleracao_y, cutoff, new_fs)

        delta_v = np.diff(velocidade_y)
        aceleracao_y = delta_v / delta_t
        aceleracao_y = np.insert(aceleracao_y, 0, 0)
        aceleracao_y_filtered = low_pass_filter(
            aceleracao_y, cutoff, new_fs)

        delta_v = np.diff(velocidade_z)
        aceleracao_z = delta_v / delta_t
        aceleracao_z = np.insert(aceleracao_z, 0, 0)
        aceleracao_z_filtered = low_pass_filter(
            aceleracao_z, cutoff, new_fs)

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

        with col2:
            # Carregar o arquivo de texto
            uploaded_file = st.file_uploader(
                "Escolha um arquivo do acelerômetro do smartphone", type=["txt"])
            # Lê o arquivo como DataFrame
            df = pd.read_csv(uploaded_file, sep=";", engine='python')

            # Verificar número de colunas e processar
            if df.shape[1] == 4:  # Verifica se há exatamente 4 colunas
                # Separar colunas
                tempo = df.iloc[:, 0].values
                acc_x = df.iloc[:, 1].values
                acc_y = df.iloc[:, 2].values
                acc_z = df.iloc[:, 3].values

                # Supõe que os dados têm uma frequência inicial uniforme
                original_fs = 50
                time_original = np.arange(0, len(tempo)) / original_fs

                # Novo eixo de tempo para interpolação (100 Hz)
                new_fs = 100
                time_interpolated = np.arange(0, time_original[-1], 1 / new_fs)

                # Interpolação
                acc_x_interpolated = interp1d(
                    time_original, acc_x, kind='linear')(time_interpolated)
                acc_y_interpolated = interp1d(
                    time_original, acc_y, kind='linear')(time_interpolated)
                acc_z_interpolated = interp1d(
                    time_original, acc_z, kind='linear')(time_interpolated)

                # Detrend
                acc_x_detrended = detrend(acc_x_interpolated)
                acc_y_detrended = detrend(acc_y_interpolated)
                acc_z_detrended = detrend(acc_z_interpolated)

                # Filtro passa-baixa (10 Hz)
                cutoff = 10  # Frequência de corte
                acc_x_filtered = low_pass_filter(
                    acc_x_detrended, cutoff, new_fs)
                acc_y_filtered = low_pass_filter(
                    acc_y_detrended, cutoff, new_fs)
                acc_z_filtered = low_pass_filter(
                    acc_z_detrended, cutoff, new_fs)

                acc_norm_filtered = np.sqrt(
                    acc_x_filtered**2+acc_y_filtered**2+acc_z_filtered**2)

                for index, value in enumerate(acc_norm_filtered):
                    if value > 2:
                        lag = time_interpolated[index]
                        time_interpolated = time_interpolated - \
                            time_interpolated[index]
                        break

                # Carregar o arquivo de texto
            if uploaded_file:
                with col3:
                    uploaded_file_gyro = st.file_uploader(
                        "Escolha o arquivo do giroscópio do smartphine", type=["txt"])
                    # Lê o arquivo como DataFrame
                    df = pd.read_csv(uploaded_file_gyro,
                                     sep=";", engine='python')

                    # Verificar número de colunas e processar
                    if df.shape[1] == 4:  # Verifica se há exatamente 4 colunas
                        # Separar colunas
                        tempo_gyro = df.iloc[:, 0].values
                        gyro_x = df.iloc[:, 1].values
                        gyro_y = df.iloc[:, 2].values
                        gyro_z = df.iloc[:, 3].values

                        # Supõe que os dados têm uma frequência inicial uniforme
                        original_fs = 50
                        time_original_gyro = np.arange(
                            0, len(tempo_gyro)) / original_fs

                        # Novo eixo de tempo para interpolação (100 Hz)
                        new_fs = 100
                        time_interpolated_gyro = np.arange(
                            0, time_original_gyro[-1], 1 / new_fs)

                        # Interpolação
                        gyro_x_interpolated = interp1d(
                            time_original_gyro, gyro_x, kind='linear')(time_interpolated_gyro)
                        gyro_y_interpolated = interp1d(
                            time_original_gyro, gyro_y, kind='linear')(time_interpolated_gyro)
                        gyro_z_interpolated = interp1d(
                            time_original_gyro, gyro_z, kind='linear')(time_interpolated_gyro)

                        # Detrend
                        gyro_x_detrended = detrend(gyro_x_interpolated)
                        gyro_y_detrended = detrend(gyro_y_interpolated)
                        gyro_z_detrended = detrend(gyro_z_interpolated)

                        # Filtro passa-baixa (10 Hz)
                        cutoff = 2  # Frequência de corte
                        gyro_x_filtered = low_pass_filter(
                            gyro_x_detrended, cutoff, new_fs)
                        gyro_y_filtered = low_pass_filter(
                            gyro_y_detrended, cutoff, new_fs)
                        gyro_z_filtered = low_pass_filter(
                            gyro_z_detrended, cutoff, new_fs)

                        gyro_norm_filtered = np.sqrt(
                            gyro_x_filtered**2+gyro_y_filtered**2+gyro_z_filtered**2)
                        #for index, value in enumerate(acc_norm_filtered):
                        #    if value > 0.1:
                        #        lag = time_interpolated_gyro[index]
                        #        time_interpolated_gyro = time_interpolated_gyro - \
                        #            time_interpolated[index]
                        #        break
                        time_interpolated_gyro = time_interpolated_gyro - time_interpolated[index]

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

                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(time_original_kinem, disp_z, 'k-')
                            ax.plot([0, 0], [0, 2], 'r-')
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
                                ax.plot(time_interpolated,
                                        acc_norm_filtered, 'k-')
                                ax.plot([0, 0], [0, 30], 'r--')
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
                                with col3:
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    ax.plot(time_interpolated_gyro,
                                            gyro_norm_filtered, 'k-')
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


               
