from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64

# Función para convertir figura de matplotlib a HTML
def plt_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}'

# Detecta las columnas del DataFrame y el y_label para el análisis.
def detectar_columnas_y_label(df):
    columnas = {
        'pre': 'RESULTADOS PRE',
        'post': 'RESULTADOS POST',
        'id1': 'IDENTIFICADOR 1'
    }
    
    # Leemos en el excel las unidades de concentración para eje Y de la gráfica
    y_label = df.iloc[0, 0]
    df = df.iloc[1:].reset_index(drop=True)
    
    return columnas, y_label, df

# Crea diferentes tipos de gráficos según la selección del usuario
def crear_visualizacion(df_long, tipo_grafico, y_label):
    plt.figure(figsize=(12, 6))
    
    if tipo_grafico == "Gráfico de cajas (Box plot)":
        sns.boxplot(data=df_long, x='Tratamiento', y='Valor', hue='Tiempo')
        plt.title('Gráfico de cajas (Box plot)', pad=10)
        plt.ylabel("Concentración en: " + y_label)
    
    elif tipo_grafico == "Gráfico de violín":
        sns.violinplot(data=df_long, x='Tratamiento', y='Valor', hue='Tiempo',
                      split=True, inner='box')
        plt.title('Gráfico de violín', pad=10)
        plt.ylabel("Concentración en: " + y_label)
        
    elif tipo_grafico == "Gráfico de puntos":
        sns.stripplot(data=df_long, x='Tratamiento', y='Valor', hue='Tiempo',
                     dodge=True, alpha=0.6)
        plt.title('Gráfico de puntos', pad=10)
        plt.ylabel("Concentración en: " + y_label)
        
    elif tipo_grafico == "Gráfico combinado (Puntos y cajas)":
        sns.boxplot(data=df_long, x='Tratamiento', y='Valor', hue='Tiempo',
                   saturation=0.7)
        
        # Puntos en negro con borde blanco para mejor visibilidad
        sns.stripplot(data=df_long, x='Tratamiento', y='Valor', hue='Tiempo',
                     dodge=True, alpha=0.7, size=5, color='black', 
                     edgecolor='white', linewidth=1)
        
        plt.title('Gráfico combinado (Puntos y cajas)', pad=10)
        plt.ylabel("Concentración en: " + y_label)
        
    elif tipo_grafico == "Gráfico de interacción":
        sns.pointplot(data=df_long, x='Tiempo', y='Valor', hue='Tratamiento',
                     errorbar=('ci', 95), capsize=0.1)
        plt.title('Gráfico de interacción', pad=10)
        plt.ylabel("Concentración en: " + y_label)
    
    # Ajusta los márgenes
    plt.tight_layout(pad=2.0)
    return plt.gcf()

def analizar_datos(archivo):
    # Configura el estilo del paquete seaborn
    sns.set_style("whitegrid")
    
    # Lee el archivo Excel
    df = pd.read_excel(archivo)
    
    # Detecta columnas y obtiene y_label
    columnas, y_label, df = detectar_columnas_y_label(df)
    
    # Limpia los datos para asegurarse de que los valores son numéricos
    for col in [columnas['pre'], columnas['post']]:
        # Limpia los espacios en blanco y convierte a numérico
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    
    # Elimina las filas con valores nulos
    df = df.dropna(subset=[columnas['pre'], columnas['post'], 'TRATAMIENTO'])
    
    # Crea identificadores únicos si no existe la columna
    if columnas['id1'] is None:
        df['ID'] = range(1, len(df) + 1)
    else:
        df['ID'] = df[columnas['id1']]
    
    # Limpia espacios en blanco en el tratamiento
    df['TRATAMIENTO'] = df['TRATAMIENTO'].str.strip()
    
    # Crea DataFrame
    df_long = pd.DataFrame({
        'Valor': pd.concat([df[columnas['pre']], df[columnas['post']]]).astype(float),
        'Tiempo': ['Pre'] * len(df) + ['Post'] * len(df),
        'Tratamiento': pd.concat([df['TRATAMIENTO'], df['TRATAMIENTO']]),
        'ID': pd.concat([df['ID'], df['ID']])
    })
    
    # Estadística descriptiva
    stats_desc = df_long.groupby(['Tratamiento', 'Tiempo'])['Valor'].describe()
    
    # ANOVA de medidas repetidas
    aov = pg.mixed_anova(data=df_long,
                        dv='Valor',
                        between='Tratamiento',
                        within='Tiempo',
                        subject='ID')
    
    # Renombra las columnas del ANOVA
    aov = aov.rename(columns={
        'Source': 'Fuente',
        'SS': 'Suma de Cuadrados',
        'DF1': 'Gl1',
        'DF2': 'Gl2',
        'MS': 'Cuadrado Medio',
        'F': 'F-razón',
        'p-unc': 'P-valor'
    })
    
    # Elimina columnas no innecesarias
    aov = aov.drop(columns=['np2', 'eps'], errors='ignore')
    
    # Crea el gráfico por defecto
    fig = crear_visualizacion(df_long, "Gráfico de cajas (Box plot)", y_label)
    
    # Pruebas t pareadas
    resultados_t = {}
    for tratamiento in df['TRATAMIENTO'].unique():
        datos_trat = df[df['TRATAMIENTO'] == tratamiento]
        t_stat, p_val = stats.ttest_rel(datos_trat[columnas['pre']], 
                                      datos_trat[columnas['post']])
        
        # Calcula el tamaño del efecto (D de Cohen)
        d = (datos_trat[columnas['pre']].mean() - datos_trat[columnas['post']].mean()) / \
            np.sqrt((datos_trat[columnas['pre']].std()**2 + datos_trat[columnas['post']].std()**2) / 2)
        
        resultados_t[tratamiento] = {
            'Estadístico-T': t_stat, 
            'P-valor': p_val,
            'D de Cohen': d
        }
    
    return aov, stats_desc, fig, resultados_t, df_long, y_label

# Define la interfaz de usuario
app_ui = ui.page_fluid(
    ui.tags.style("""
        /* Variables CSS */
        :root {
            --primary-color: #2193b0;
            --secondary-color: #6dd5ed;
            --bg-color: #f5f7fa;
            --text-color: #2c3e50;
            --card-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            --transition-speed: 0.3s;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #ebf0f6 100%);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: var(--text-color);
            line-height: 1.6;
            animation: fadeIn 0.8s ease-out;
        }

        .encabezado {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0 30px 0;
            box-shadow: var(--card-shadow);
            position: relative;
            overflow: hidden;
            transition: all var(--transition-speed) ease;
            border: 1px solid black;
        }

        .info-autor {
            color: rgba(255, 255, 255, 0.9);
            font-size: 0.95rem;
            margin-top: 15px;
            display: flex;
            gap: 20px;
            opacity: 0.85;
            font-weight: 500;
            padding: 5px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        .info-autor span {
            display: inline-flex;
            align-items: center;
        }

        .info-autor span::before {
            content: '•';
            margin-right: 8px;
            color: rgba(255, 255, 255, 0.7);
        }

        .info-autor span:first-child::before {
            display: none;
        }

        .encabezado h2 {
            margin: 0;
            font-weight: 600;
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .caja {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: var(--card-shadow);
            transition: all var(--transition-speed) cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            border: 1px solid black;
        }

        .caja h3 {
            color: var(--primary-color);
            font-size: 1.4rem;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(33, 147, 176, 0.1);
            transition: color var(--transition-speed) ease;
        }

        .panel-control {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: var(--card-shadow);
            transition: all var(--transition-speed) ease;
            border: 1px solid black;
        }

        .input-fichero-contenedor {
            margin-bottom: 25px;
            position: relative;
            padding: 10px;
            border-radius: 8px;
        }

        input[type="file"] {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 2px dashed rgba(33, 147, 176, 0.3);
            transition: all var(--transition-speed) ease;
        }

        input[type="file"]:hover {
            border-color: var(--primary-color);
            background: rgba(33, 147, 176, 0.05);
        }

        input[type="file"]::file-selector-button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            margin-right: 15px;
        }

        input[type="file"]::file-selector-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(33, 147, 176, 0.3);
        }

        select {
            width: 100%;
            padding: 12px;
            border: 2px solid rgba(33, 147, 176, 0.2);
            border-radius: 8px;
            background-color: white;
            color: var(--text-color);
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232193b0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 10px center;
            background-size: 20px;
        }

        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 15px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease-out;
            border: 1px solid black;
        }

        th {
            background: linear-gradient(to right, #f8fafc, #f1f5f9);
            color: var(--text-color);
            font-weight: 600;
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #e1e8ed;
            position: relative;
        }

        td {
            padding: 15px;
            border-bottom: 1px solid #e1e8ed;
            transition: all var(--transition-speed) ease;
        }

        tr:hover td {
            background-color: rgba(33, 147, 176, 0.05);
        }

        .contenedor-grafico {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: var(--card-shadow);
            margin: 15px 0;
            min-height: 600px;
            transition: all var(--transition-speed) ease;
            border: 1px solid black;
        }

        .visualizacion-caja {
            padding-bottom: 35px;
        }

        .notification {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            animation: slideInNotification 0.5s ease;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
            border: 1px solid black;
        }

    """),
    
    # Título
    ui.div(
        {"class": "encabezado"},
        ui.h2("Análisis Estadístico de Datos"),
        ui.div(
            {"class": "info-autor"},
            ui.span("Autor: Ignacio González Arnaiz"),
            ui.span("Titulación: Ingeniería de la Salud UBU")
        )
    ),
    
    # Interfaz
    ui.row(
        # Panel izquierdo (selección)
        ui.column(3,
            ui.div(
                {"class": "panel-control"},
                ui.h3("Controles"),
                ui.div(
                    {"class": "input-fichero-contenedor"},
                    ui.input_file("file", "Seleccionar archivo Excel", 
                                accept=[".xlsx"], multiple=False)
                ),
                ui.input_select(
                    "tipo_grafico",
                    "Tipo de Gráfico",
                    choices=[
                        "Gráfico de cajas (Box plot)",
                        "Gráfico de violín",
                        "Gráfico de puntos",
                        "Gráfico combinado (Puntos y cajas)",
                        "Gráfico de interacción"
                    ]
                )
            )
        ),
        
        # Panel derecho (resultados)
        ui.column(9,
            ui.div(
                {"class": "caja"},
                ui.h3("Estadística Descriptiva"),
                ui.output_ui("tabla_estadisticas")
            ),
            
            ui.div(
                {"class": "caja"},
                ui.h3("Resultados del ANOVA de Medidas Repetidas"),
                ui.output_ui("tabla_anova")
            ),
            
            ui.div(
                {"class": "caja"},
                ui.h3("Prueba T Pareada"),
                ui.output_ui("tabla_ttest")
            ),
            
            ui.div(
                {"class": "caja visualizacion-caja"},
                ui.h3("Visualizaciones"),
                ui.div(
                    {"class": "contenedor-grafico"},
                    ui.output_plot("plot", width="100%", height="550px")
                )
            )
        )
    )
)

def server(input, output, session):
    results = reactive.Value()     # Almacena resultados de tablas (ANOVA, t-test, etc.)
    df_long = reactive.Value()     # Almacena el dataframe
    y_label = reactive.Value()     # Almacena la etiqueta del eje Y para los gráficos
    
    # Procesa el archivo Excel cuando se sube uno nuevo, realiza el análisis estadístico y actualiza las variables reactivas y muestra un mensaje de éxito o de error
    @reactive.Effect
    @reactive.event(input.file)
    def process_file():
        if input.file() is not None:
            file_infos = input.file()
            try:
                # Realiza el análisis
                aov, stats_desc, fig, res_t, df, label = analizar_datos(file_infos[0]['datapath'])
                
                # Convierte los resultados a HTML
                stats_desc_html = stats_desc.to_html(classes='table table-striped', 
                                                   float_format=lambda x: '%.3f' % x if pd.notnull(x) else '')
                aov_html = aov.to_html(classes='table table-striped', 
                                     float_format=lambda x: '%.4f' % x if pd.notnull(x) else '')
                
                df_t = pd.DataFrame.from_dict(res_t, orient='index')
                df_t.columns = ['Estadistico-T', 'P-valor', "D de Cohen"]
                t_test_html = df_t.to_html(classes='table table-striped', 
                                         float_format=lambda x: '%.4f' % x if pd.notnull(x) else '')
                
                # Guarda los resultados y datos
                results.set({
                    'aov_html': aov_html,
                    'stats_desc_html': stats_desc_html,
                    't_test_html': t_test_html
                })

                df_long.set(df)
                y_label.set(label)
                
                ui.notification_show("Análisis completado con éxito!", type="message")
                
            except Exception as e:
                print(f"Error detallado: {str(e)}")
                ui.notification_show(f"Error en el análisis: {str(e)}", type="error")
    
    # Renderiza la tabla de estadísticas descriptivas y muestra un mensaje si no hay archivo subido
    @output
    @render.ui
    def tabla_estadisticas():
        if results() is not None:
            return ui.HTML(results()['stats_desc_html'])
        return ui.p("Sube un archivo Excel para ver los resultados")
    
    # Renderiza la tabla de resultados del ANOVA y muestra un mensaje si no hay archivo subido
    @output
    @render.ui
    def tabla_anova():
        if results() is not None:
            return ui.HTML(results()['aov_html'])
        return ui.p("Sube un archivo Excel para ver los resultados")
    
    # Renderiza la tabla de resultados del t-test pareado y muestra un mensaje si no hay archivo subido
    @output
    @render.ui
    def tabla_ttest():
        if results() is not None:
            return ui.HTML(results()['t_test_html'])
        return ui.p("Sube un archivo Excel para ver los resultados")
    
    # Renderiza el gráfico seleccionado y se actualiza cuando cambia el archivo o el tipo de gráfico
    @output
    @render.plot
    @reactive.event(input.file, input.tipo_grafico)
    def plot():
        if df_long() is not None and y_label() is not None:
            return crear_visualizacion(df_long(), input.tipo_grafico(), y_label())
        return None

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()