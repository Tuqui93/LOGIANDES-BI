"""LogiAndes S.A. — Monitor de inteligencia de negocios.

Arranque: python -m streamlit run LogiAndes_BI_Integrado.py
Entradas: pedidos_para_bi.csv (obligatorio), metas_escenario.csv (opcional),
          fichas_kpi.csv, resumen_provincias.csv, plan_accion.csv (opcionales).
Los archivos pueden estar en la misma carpeta, en salida_logiandes/ o en un ZIP.
"""
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='LogiAndes S.A. | Inteligencia de Negocios', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent
FILES = ['pedidos_para_bi.csv','metas_escenario.csv','fichas_kpi.csv','resumen_provincias.csv','plan_accion.csv','respuestas_areas.csv','resumen_provincia_mes.csv','costos_supuestos.csv']
NECESARIOS = {'pedido_id','fecha_pedido','provincia','canal_entrega','ventas_asociadas_usd','hubo_reclamo','tiempo_individual_entrega_min','costo_directo_estimado_usd','margen_bruto_estimado_usd'}

@st.cache_data(show_spinner=False)
def csv_bytes(content):
    return pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')

@st.cache_data(show_spinner=False)
def unzip_csv(content):
    out = {}
    with ZipFile(io.BytesIO(content)) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if name in FILES and not info.is_dir():
                if info.file_size > 30_000_000:
                    raise ValueError(f'El archivo {name} excede 30 MB.')
                out[name] = z.read(info)
    return out

def cargar_archivos():
    st.sidebar.markdown('### Origen de información')
    st.sidebar.caption('Carga un ZIP exportado del procesamiento o deja los CSV junto a este script.')
    paquete = st.sidebar.file_uploader('Paquete de resultados (.zip)',type=['zip'])
    subido = st.sidebar.file_uploader('CSV de pedidos (opcional)',type=['csv'])
    meta_file = st.sidebar.file_uploader('CSV de metas (opcional)',type=['csv'])
    archivos = {}
    for filename in FILES:
        for p in (ROOT / filename, ROOT / 'salida_logiandes' / filename):
            if p.is_file():
                archivos[filename] = p.read_bytes()
                break
    if paquete:
        try: archivos.update(unzip_csv(paquete.getvalue()))
        except (BadZipFile, ValueError) as e:
            st.error(f'No se pudo abrir el ZIP: {e}')
            st.stop()
    if subido: archivos['pedidos_para_bi.csv'] = subido.getvalue()
    if meta_file: archivos['metas_escenario.csv'] = meta_file.getvalue()
    if 'pedidos_para_bi.csv' not in archivos:
        st.info('Para comenzar, carga **pedidos_para_bi.csv** o el ZIP de resultados generado en el análisis.')
        st.stop()
    tablas = {}
    for nombre, data in archivos.items():
        try: tablas[nombre] = csv_bytes(data)
        except Exception as e:
            st.warning(f'El archivo {nombre} no pudo leerse: {e}')
    st.sidebar.caption(f'Archivos disponibles: {len(tablas)}')
    return tablas

def validar(tablas):
    df = tablas['pedidos_para_bi.csv'].copy()
    faltan = NECESARIOS.difference(df.columns)
    if faltan: st.error('Faltan columnas obligatorias: '+', '.join(sorted(faltan))); st.stop()
    if df['pedido_id'].isna().any() or df['pedido_id'].duplicated().any():
        st.error('Hay pedidos con identificador vacío o duplicado; revisa la calidad de la base.'); st.stop()
    df['fecha_pedido'] = pd.to_datetime(df['fecha_pedido'], errors='coerce')
    if df['fecha_pedido'].isna().any(): st.error('Se encontraron fechas inválidas.'); st.stop()
    df['mes'] = df['fecha_pedido'].dt.strftime('%Y-%m')
    for c in ['ventas_asociadas_usd','hubo_reclamo','tiempo_individual_entrega_min','costo_directo_estimado_usd','margen_bruto_estimado_usd']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[list(NECESARIOS.intersection(df.columns).difference({'pedido_id','fecha_pedido','provincia','canal_entrega'}))].isna().any().any():
        st.error('Hay indicadores numéricos inválidos o vacíos.'); st.stop()
    if not df['hubo_reclamo'].isin([0,1]).all(): st.error('hubo_reclamo debe ser 0 o 1.'); st.stop()
    df['provincia'] = df['provincia'].astype(str).str.strip()
    df['canal_entrega'] = df['canal_entrega'].astype(str).str.strip()
    if not np.allclose(df.ventas_asociadas_usd-df.costo_directo_estimado_usd,df.margen_bruto_estimado_usd,atol=.02):
        st.warning('Algunos márgenes no coinciden con ventas menos costos; revisar el archivo de origen.')
    metas = tablas.get('metas_escenario.csv')
    if metas is not None:
        if not {'provincia','mes','meta_ventas_usd'}.issubset(metas.columns):
            st.warning('La tabla de metas carece de columnas obligatorias; se omitirán las metas.'); metas=None
        else:
            metas=metas.copy()
            metas['provincia']=metas.provincia.astype(str).str.strip()
            metas['mes']=metas.mes.astype(str).str.strip()
            metas['meta_ventas_usd']=pd.to_numeric(metas.meta_ventas_usd,errors='coerce')
            if metas.duplicated(['provincia','mes']).any() or metas.meta_ventas_usd.isna().any() or (metas.meta_ventas_usd<=0).any():
                st.warning('Metas duplicadas, vacías o no positivas; se omitirán las metas.'); metas=None
    return df, metas

def dinero(n): return f'US$ {n:,.2f}'
def tasa(n): return f'{n:,.1f}'

def indicadores(x):
    n=x.pedido_id.nunique()
    v=x.ventas_asociadas_usd.sum(); c=x.costo_directo_estimado_usd.sum()
    return {'pedidos':n,'ventas':v,'costos':c,'margen_usd':v-c,'margen_pct':100*(v-c)/v if v else np.nan,
            'reclamos':int(x.hubo_reclamo.sum()),'reclamos_x1000':100*x.hubo_reclamo.sum()*10/n if n else np.nan,
            'promedio_min':x.tiempo_individual_entrega_min.mean(), 'p90_min':x.tiempo_individual_entrega_min.quantile(.90)}

def resumen(x, campo):
    g=x.groupby(campo,dropna=False)
    r=g.agg(pedidos=('pedido_id','nunique'),ventas=('ventas_asociadas_usd','sum'),costos=('costo_directo_estimado_usd','sum'),
            reclamos=('hubo_reclamo','sum'),tiempo_promedio=('tiempo_individual_entrega_min','mean'),
            tiempo_mediana=('tiempo_individual_entrega_min','median'),p90=('tiempo_individual_entrega_min',lambda v:v.quantile(.9)),
            desv=('tiempo_individual_entrega_min','std')).reset_index()
    r['margen_pct']=100*(r.ventas-r.costos)/r.ventas.replace(0,np.nan)
    r['reclamos_x1000']=1000*r.reclamos/r.pedidos
    r['cv_pct']=100*r.desv/r.tiempo_promedio.replace(0,np.nan)
    return r

def cuadro(fig,key,height=385):
    fig.update_layout(height=height,margin=dict(l=8,r=10,t=42,b=15),legend_title_text='',hovermode='closest')
    st.plotly_chart(fig,use_container_width=True,key=key,config={'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})

def estado(x,verde,amarillo,menor=False):
    if x is None or pd.isna(x): return 'Sin dato'
    if menor: return 'Óptimo' if x<=verde else 'Atención' if x<=amarillo else 'Crítico'
    return 'Óptimo' if x>=verde else 'Atención' if x>=amarillo else 'Crítico'

def main():
    tablas=cargar_archivos()
    df,metas=validar(tablas)
    st.sidebar.markdown('### Presentación')
    estilo=st.sidebar.radio('Estilo de navegación',['Oscuro Neón','Cápsula Clara','Rojo Vibrante'],index=0)
    colores={'Oscuro Neón':('#18181b','#f43f5e','#f4f4f5'), 'Cápsula Clara':('#f1f5f9','#d90429','#0f172a'), 'Rojo Vibrante':('#b91c1c','#ffffff','#ffffff')}
    bg,accent,fg=colores[estilo]
    st.markdown(f'''<style>.block-container{{padding-top:1.2rem;max-width:1500px}} div[data-testid="stTabs"]>div[role="tablist"]{{background:{bg};padding:8px;border-radius:20px;gap:8px;flex-wrap:wrap}}div[data-testid="stTabs"] button[role="tab"]{{border-radius:12px;padding:8px 15px;color:{fg}}} div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{{box-shadow:inset 0 -3px 0 {accent};font-weight:800}} </style>''',unsafe_allow_html=True)
    st.title('LogiAndes S.A. | Inteligencia de Negocios')
    st.caption('Seguimiento comercial, territorial y logístico · Datos observados y escenario de planificación diferenciados')
    st.sidebar.markdown('### Filtros globales')
    fechas=sorted(df.mes.unique());provs=sorted(df.provincia.unique());canales=sorted(df.canal_entrega.unique())
    meses=st.sidebar.multiselect('Periodo (mes)',fechas,default=fechas)
    provincias=st.sidebar.multiselect('Provincia',provs,default=provs)
    canales_sel=st.sidebar.multiselect('Canal',canales,default=canales)
    d=df[df.mes.isin(meses)&df.provincia.isin(provincias)&df.canal_entrega.isin(canales_sel)].copy()
    if d.empty: st.warning('No existen registros con los filtros elegidos.');st.stop()
    k=indicadores(d)
    filtra_canal=set(canales_sel)!=set(canales)
    meta_total=None; meta_mes=None
    if metas is not None:
        claves=d[['provincia','mes']].drop_duplicates()
        ver=claves.merge(metas,on=['provincia','mes'],how='left',validate='one_to_one')
        if ver.meta_ventas_usd.isna().any(): st.warning('Faltan metas para algunos territorios/meses; cumplimiento no disponible.')
        else:
            meta_total=ver.meta_ventas_usd.sum()
            meta_mes=ver.groupby('mes',as_index=False).meta_ventas_usd.sum()
    # Meta formal corresponde al territorio y periodo completos, no se fracciona artificialmente por canal.
    cumplimiento=None if filtra_canal or not meta_total else 100*k['ventas']/meta_total
    st.caption(f'Cobertura seleccionada: {k["pedidos"]:,} pedidos | {len(provincias)} provincias seleccionadas | {len(meses)} meses')
    if filtra_canal: st.info('El filtro de canal está activo. No se informa cumplimiento de metas territoriales, porque no existen presupuestos desagregados por canal.')
    tabs=st.tabs(['1. Visión Ejecutiva','2. Diagnóstico Territorial','3. Detalle Operativo','4. Ficha BI y Gobernanza'])
    with tabs[0]:
        st.subheader('Indicadores prioritarios')
        a,b,c,e,f=st.columns(5)
        a.metric('Ventas asociadas',dinero(k['ventas']),help='No se pueden certificar como ventas netas contables.')
        b.metric('Margen bruto estimado',f'{k["margen_pct"]:.2f}%',help=f'Sobre costos modelados: {dinero(k["margen_usd"])} de margen monetario.')
        c.metric('Cumplimiento de meta',f'{cumplimiento:.2f}%' if cumplimiento is not None else 'N/D',help='Comparación con metas simuladas. No disponible si se filtra el canal.')
        e.metric('Reclamos por 1.000',f'{k["reclamos_x1000"]:.2f}',help=f'{k["reclamos"]} pedidos con reclamo')
        f.metric('Tiempo medio',f'{k["promedio_min"]:.1f} min',help=f'Percentil 90: {k["p90_min"]:.1f} minutos.')
        st.caption('Ventas y tiempos: observados. Margen y cumplimiento: escenario de costos y metas; no representan contabilidad certificada. El OTD no es calculable sin tiempos comprometidos.')
        c1,c2=st.columns(2)
        r=resumen(d,'provincia')
        with c1:
            fig=px.bar(r.sort_values('ventas'),x='ventas',y='provincia',orientation='h',text='ventas',color='ventas',color_continuous_scale='Blues',
                       hover_data={'pedidos':True,'margen_pct':':.2f','reclamos_x1000':':.2f'},labels={'ventas':'Ventas asociadas (USD)','provincia':'Provincia'},title='Ventas asociadas por provincia')
            fig.update_traces(texttemplate='%{text:,.0f}',textposition='outside');fig.update_layout(coloraxis_showscale=False)
            cuadro(fig,'exec_ventas')
        with c2:
            actual=d.groupby('mes',as_index=False).ventas_asociadas_usd.sum()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=actual.mes,y=actual.ventas_asociadas_usd,mode='lines+markers',name='Ventas asociadas',hovertemplate='%{x}: US$ %{y:,.2f}<extra></extra>'))
            if meta_mes is not None:
                fig.add_trace(go.Scatter(x=meta_mes.mes,y=meta_mes.meta_ventas_usd,mode='lines+markers',name='Meta simulada provincial',line={'dash':'dash'}))
            fig.update_layout(title='Evolución mensual frente a metas de escenario',yaxis_title='USD')
            if filtra_canal: st.caption('Las metas corresponden al total de cada provincia, no al canal seleccionado.')
            cuadro(fig,'exec_serie')
        c3,c4=st.columns(2)
        with c3:
            fig=px.bar(r.sort_values('reclamos_x1000'),x='reclamos_x1000',y='provincia',orientation='h',color='reclamos_x1000',
                       color_continuous_scale='Reds',text='reclamos_x1000',hover_data=['reclamos','pedidos'],title='Reclamos por cada 1.000 pedidos')
            fig.add_vline(x=25,line_dash='dash',annotation_text='Referencia ilustrativa ≤25')
            fig.update_layout(coloraxis_showscale=False,xaxis_title='Reclamos por 1.000',yaxis_title='');cuadro(fig,'exec_reclamos')
        with c4:
            fig=px.bar(r.sort_values('margen_pct'),x='margen_pct',y='provincia',orientation='h',color='margen_pct',color_continuous_scale='RdYlGn',
                       text='margen_pct',hover_data=['ventas','costos'],title='Margen bruto estimado por provincia')
            fig.add_vline(x=25,line_dash='dash',annotation_text='Referencia ilustrativa 25%');fig.update_layout(coloraxis_showscale=False,xaxis_title='% de margen estimado',yaxis_title='');cuadro(fig,'exec_margen')
        if cumplimiento is not None:
            sem=estado(cumplimiento,95,85);st.write(f'**Seguimiento de meta simulada:** {cumplimiento:.2f}% · Estado orientativo: {sem}.')
        st.divider()
        r_ventas=r.loc[r.ventas.idxmax()];r_recl=r.loc[r.reclamos_x1000.idxmax()];r_tiempo=r.loc[r.tiempo_promedio.idxmax()]
        st.info(f'**Lectura de gestión:** {r_ventas.provincia} aporta las mayores ventas del filtro; {r_recl.provincia} registra la mayor tasa de reclamos ({r_recl.reclamos_x1000:.1f} por 1.000); {r_tiempo.provincia} muestra la mayor duración media ({r_tiempo.tiempo_promedio:.1f} min). Son señales para investigar, no pruebas de causalidad.')
    with tabs[1]:
        st.subheader('Composición y comparación territorial')
        st.caption('Los indicadores de reclamación se expresan como tasas comparables; el volumen de pedidos se conserva como contexto.')
        if 'ciudad' not in d.columns or 'linea_servicio' not in d.columns:
            st.caption('El origen actual no registra ciudad ni línea de servicio; no se fabrican estas dimensiones.')
        metrica=st.radio('Variable de participación',['Ventas asociadas','Pedidos'],horizontal=True)
        figura=st.radio('Presentación',['Vista integrada','Sunburst','Mapa de calor','Dispersión','Tabla'],horizontal=True)
        valor='ventas_asociadas_usd' if metrica=='Ventas asociadas' else None
        def sunburst():
            if valor: fig=px.sunburst(d[d.ventas_asociadas_usd>0],path=['provincia','canal_entrega'],values=valor,title='Participación por provincia y canal')
            else: fig=px.sunburst(d,path=['provincia','canal_entrega'],title='Participación de pedidos por provincia y canal')
            fig.update_traces(textinfo='label+percent parent');cuadro(fig,'territ_sun',420)
        def heatmap():
            pv=d.pivot_table(index='provincia',columns='canal_entrega',values='ventas_asociadas_usd',aggfunc='sum',fill_value=0) if valor else pd.crosstab(d.provincia,d.canal_entrega)
            fig=px.imshow(pv,aspect='auto',text_auto='.2s',color_continuous_scale='Blues',title=f'Matriz provincia × canal ({metrica.lower()})')
            cuadro(fig,'territ_heat',420)
        def scatter():
            r=resumen(d,'provincia')
            fig=px.scatter(r,x='pedidos',y='reclamos_x1000',size='ventas',color='tiempo_promedio',text='provincia',size_max=54,
                           hover_data={'ventas':':,.2f','reclamos':True,'margen_pct':':.2f','p90':':.1f','tiempo_promedio':':.1f'},
                           labels={'pedidos':'Cantidad de pedidos','reclamos_x1000':'Reclamos por 1.000 pedidos','tiempo_promedio':'Tiempo medio (min)'},
                           title='Volumen frente a tasa de reclamos (con contexto comercial)')
            fig.update_traces(textposition='top center');cuadro(fig,'territ_scatter',450)
        if figura=='Vista integrada':
            c1,c2=st.columns(2)
            with c1: sunburst()
            with c2: heatmap()
            scatter()
        elif figura=='Sunburst': sunburst()
        elif figura=='Mapa de calor': heatmap()
        elif figura=='Dispersión': scatter()
        if figura in ('Vista integrada','Tabla'):
            st.markdown('**Comparación provincial de indicadores**')
            st.dataframe(resumen(d,'provincia').round(2).sort_values('ventas',ascending=False),hide_index=True,use_container_width=True)
    with tabs[2]:
        st.subheader('Variabilidad logística y control del servicio')
        st.caption('Una entrega larga no es necesariamente tardía; falta la promesa contractual para calcular OTD.')
        filtro=st.selectbox('Segmentar distribución',['provincia','canal_entrega'],format_func=lambda x:'Provincia' if x=='provincia' else 'Canal de entrega')
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(d,x=filtro,y='tiempo_individual_entrega_min',color=filtro,points='outliers',
                       hover_data=['pedido_id','provincia','canal_entrega'],title='Distribución y casos atípicos de entrega')
            fig.update_layout(showlegend=False,yaxis_title='Minutos',xaxis_title='');cuadro(fig,'oper_box')
        with c2:
            r=resumen(d,filtro).sort_values('p90')
            fig=px.bar(r,x='p90',y=filtro,orientation='h',color='cv_pct',text='p90',
                       hover_data=['pedidos','tiempo_promedio','cv_pct'],title='P90: duración bajo la cual se completa el 90%')
            fig.update_traces(texttemplate='%{text:.1f}',textposition='outside');fig.update_layout(xaxis_title='Minutos');cuadro(fig,'oper_p90')
        r=resumen(d,filtro)
        st.dataframe(r[[filtro,'pedidos','tiempo_promedio','tiempo_mediana','p90','desv','cv_pct','reclamos_x1000']].sort_values('cv_pct',ascending=False).round(2),hide_index=True,use_container_width=True)
        st.markdown('**Exploración operativa de pedidos**')
        incid=st.selectbox('Mostrar',['Todos los pedidos','Solo pedidos con reclamo','Pedidos sin reclamo'])
        detalle=d if incid=='Todos los pedidos' else d[d.hubo_reclamo.eq(1 if incid=='Solo pedidos con reclamo' else 0)]
        campos=['pedido_id','fecha_pedido','provincia','canal_entrega','tipo_cliente','tiempo_individual_entrega_min','reclamos_registrados','motivo_reclamo','ventas_asociadas_usd']
        campos=[c for c in campos if c in detalle]
        st.dataframe(detalle[campos],height=320,hide_index=True,use_container_width=True)
        st.download_button('Descargar detalle filtrado',detalle[campos].to_csv(index=False).encode('utf-8-sig'),'logiandes_detalle_filtrado.csv','text/csv')
        st.warning('El detalle usa únicamente campos operativos; no se habilitan datos personales mediante un interruptor sin autenticación.')
    with tabs[3]:
        st.subheader('Definición de indicadores y gobernanza')
        ficha=tablas.get('fichas_kpi.csv')
        if ficha is not None: st.dataframe(ficha,hide_index=True,use_container_width=True)
        else:
            st.markdown('''- **Ventas asociadas:** suma de importes registrados en pedidos, sin validación de ventas netas.\n- **Margen estimado:** (ventas asociadas - costos directos estimados) / ventas asociadas × 100.\n- **Cumplimiento:** ventas / meta provincial-mensual simulada × 100 (solo periodos y canales completos).\n- **Reclamos por 1.000:** pedidos con reclamo / total de pedidos × 1.000.\n- **Tiempo medio y P90:** características observadas de la duración.\n- **OTD:** no calculable sin una promesa de entrega registrada.''')
        with st.expander('Calidad y trazabilidad de la base'):
            st.write(f'Pedidos únicos: **{df.pedido_id.nunique():,}**; fechas: **{df.fecha_pedido.min():%d/%m/%Y}** a **{df.fecha_pedido.max():%d/%m/%Y}**.')
            quality=pd.DataFrame({'campo':df.columns,'nulos':df.isna().sum().values,'nulos_pct':(100*df.isna().mean()).round(2).values})
            st.dataframe(quality,hide_index=True,use_container_width=True)
            st.caption('Los valores ausentes en motivo_reclamo pueden corresponder a pedidos sin reclamo.')
        for titulo,filename in [('Respuestas por área','respuestas_areas.csv'),('Plan de acción propuesto','plan_accion.csv'),('Supuestos de costo por canal','costos_supuestos.csv')]:
            t=tablas.get(filename)
            if t is not None:
                st.markdown(f'**{titulo}**');st.dataframe(t,hide_index=True,use_container_width=True)
        st.markdown('''**Criterios de interpretación:** las diferencias territoriales orientan investigaciones; no demuestran causalidad. Las metas y costos son escenarios ilustrativos, no presupuesto ni contabilidad aprobados. La meta territorial no debe sumarse una vez por pedido y tampoco asignarse arbitrariamente a cada canal. La información financiera oficial requiere conciliación con registros reales.\n\n**Acceso y privacidad:** la publicación no implementa autenticación empresarial ni seguridad por rol. No cargar datos personales identificables. Los filtros de ciudad y línea de servicio solo pueden habilitarse con registros reales de esas dimensiones.\n\n**Frecuencia recomendada:** revisión operativa semanal, seguimiento ejecutivo mensual, ajustes documentados de metas y definiciones cuando sean aprobados.''')
        st.divider()
        st.markdown('**Exportación para conciliación**')
        st.download_button('Exportar resumen territorial',resumen(d,'provincia').to_csv(index=False).encode('utf-8-sig'),'logiandes_resumen_dashboard.csv','text/csv')
    st.divider()
    st.caption('LogiAndes S.A. · Prototipo BI: información observada y supuestos de planificación diferenciados.')

if __name__=='__main__': main()
