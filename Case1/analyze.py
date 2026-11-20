import sys, os, pandas as pd, numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'students.csv'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anomalies.csv')
CODE = {'Проверка заявки':'C','Согласование руководителем':'M','Согласование бюджета':'B',
        'Проверка договора':'L','Исполнение заявки':'E','Доработка заявки':'R','Внеплановая проверка':'A'}
NAME = {v:k for k,v in CODE.items()}
EXPROLE = {'C':'specialist','M':'manager','B':'finance','L':'legal','E':'procurement','R':'requester','A':'auditor'}
CUT = pd.Timestamp('2026-07-01 00:00', tz='Etc/GMT-3')

def work_hours(a, b):
    """Рабочие часы пн–пт 09:00–18:00 между a и b."""
    if b <= a: return 0.0
    tot, d = 0.0, a.normalize()
    while d <= b:
        if d.dayofweek < 5:
            s, e = max(a, d+pd.Timedelta(hours=9)), min(b, d+pd.Timedelta(hours=18))
            if e > s: tot += (e-s).total_seconds()/3600
        d += pd.Timedelta(days=1)
    return tot

# ---------- 1. загрузка и контроль качества ----------
df = pd.read_csv(SRC, encoding='utf-8-sig', dtype=str)
df['EventIndex'] = df.EventIndex.astype(int); df['Amount'] = df.Amount.astype(float)
df['ts'] = pd.to_datetime(df.Timestamp, utc=True).dt.tz_convert('Etc/GMT-3')
df = df.sort_values(['ProcessInstanceId','EventIndex']).reset_index(drop=True)
OBS_END = max(CUT, df.ts.max())   # в выгрузке есть события ПОСЛЕ заявленного среза 00:00 01.07 -> реальный конец данных
print('ВНИМАНИЕ: событий позже заявленного среза:', (df.ts >= CUT).sum(), '| последнее событие', df.ts.max())
print('строк', len(df), 'процессов', df.ProcessInstanceId.nunique(),
      '| дубли EventId', df.EventId.duplicated().sum(), '| событий после среза', (df.ts >= CUT).sum(),
      '| TemplateVersion', df.TemplateVersion.unique())

# ---------- 2. таблица заданий (created + completed по TaskId) ----------
cr = df[df.EventType=='task_created'].set_index('TaskId')
co = df[df.EventType=='task_completed'].set_index('TaskId')
t = cr[['ProcessInstanceId','EventIndex','ActivityName','ts','RequestType']].join(
    co[['ts','UserId','Role','Result']], rsuffix='_c').reset_index()
t['code'] = t.ActivityName.map(CODE)
t['sec'] = (t.ts_c - t.ts).dt.total_seconds()
t['wh'] = [work_hours(a, b) if pd.notna(b) else np.nan for a, b in zip(t.ts, t.ts_c)]
t = t.sort_values(['ProcessInstanceId','EventIndex'])

starter = df[df.EventType=='process_started'].set_index('ProcessInstanceId')
rtype = starter.RequestType
rows = []
def add(pid, typ, expl): rows.append((pid, typ, expl))

# ---------- 3. правила по процессам ----------
for pid, g in t.groupby('ProcessInstanceId'):
    seq = ''.join(g.code)                      # порядок создания заданий
    rt = rtype[pid]; ev = df[df.ProcessInstanceId==pid]
    done = g[g.ts_c.notna()]
    res = ''.join(done.code)
    ret = (g.Result=='return').sum()
    # (а) нет согласования руководителем при успешном исполнении
    if 'E' in seq and 'M' not in seq:
        add(pid,'Пропущено согласование руководителем',
            f'Маршрут {" > ".join(NAME[c] for c in seq)} ({rt}): этап «Согласование руководителем» отсутствует, '
            f'заявка дошла до бюджета и исполнения. У всех остальных заявок, дошедших до бюджета, этап руководителя есть.')
    # (б) исполнение раньше бюджета / юриста
    if 'E' in seq and 'B' in seq and seq.index('E') < seq.index('B'):
        extra = ''
        if 'L' in seq and seq.index('E') < seq.index('L'):
            extra = ' Юридическая проверка также выполнена после исполнения.'
        e = g[g.code=='E'].iloc[0]; b = g[g.code=='B'].iloc[0]
        add(pid,'Исполнение до проверки бюджета',
            f'Порядок этапов {" > ".join(seq)}: «Исполнение заявки» создано {e.ts:%d.%m.%Y %H:%M} раньше «Согласования бюджета» ({b.ts:%d.%m.%Y %H:%M}); '
            f'по схеме бюджет (и юрист для contract) должен предшествовать исполнению.{extra} Тип заявки: {rt}.')
    # (в) дублирующая проверка заявки без возврата
    if seq.startswith('CC'):
        c = g[g.code=='C']
        add(pid,'Повторная проверка без возврата',
            f'Этап «Проверка заявки» выполнен дважды подряд (задания {c.ts.iloc[0]:%d.%m %H:%M} и {c.ts.iloc[1]:%d.%m %H:%M}) без возврата руководителем и без доработки; '
            f'в норме проверка идёт один раз до руководителя или повторно только после доработки.')
    # (г) внеплановые проверки
    if 'A' in seq:
        a = g[g.code=='A']
        add(pid,'Внеплановая проверка (auditor)',
            f'В процессе {len(a)} задания «Внеплановая проверка» (auditor), порядок {" > ".join(seq)} ({rt}). '
            f'Такие проверки есть лишь у 8 заявок из 1000, оправданность по данным не установить — нужна ручная проверка причины. Не обязательно нарушение.')
    # (д) избыточные возвраты
    if ret >= 5:
        add(pid,'Избыточное число возвратов',
            f'{ret} возвратов руководителем и столько же доработок (норма: 1 возврат у 119 из 127 заявок с возвратом; здесь {ret}), тип {rt}. '
            f'Цикл «руководитель → доработка → проверка» повторяется, заявка прошла {len(g)} заданий.')
    # (е) руководитель = requester
    m = g[(g.code=='M') & (g.Role.notna()) & (g.Role!='manager')]
    for _, r in m.iterrows():
        same = ' Пользователь совпадает с инициатором заявки (самосогласование).' if r.UserId == starter.UserId[pid] else ''
        add(pid,'Согласование выполнено не руководителем',
            f'Задание «Согласование руководителем» закрыто пользователем с ролью {r.Role} (ожидается manager), результат {r.Result}, {r.ts_c:%d.%m.%Y %H:%M}.{same}')
    # (ж) незакрытое согласование — зависание
    for _, r in g[g.ts_c.isna()].iterrows():
        if r.code == 'M':
            add(pid,'Зависшее согласование руководителем',
                f'Задание «Согласование руководителем» создано {r.ts:%d.%m.%Y %H:%M} и не закрыто к концу выгрузки (последнее событие 01.07.2026 17:59): ожидание ≈ {work_hours(r.ts,OBS_END):.0f} рабочих часов '
                f'({(OBS_END-r.ts).days} календарных дней) при медиане закрытия этапа ≈ 2 рабочих часов (максимум у закрытых — 6.8 ч). Процесс не завершён, не продвигается с этого этапа.')
    # (з) мгновенное прохождение
    if len(done) >= 2 and (done.sec < 120).sum() >= 2 and (done.sec < 120).sum() == len(done):
        s = ', '.join(f'{NAME[c]} {x:.0f} с' for c, x in zip(done.code, done.sec))
        add(pid,'Мгновенное закрытие всех заданий',
            f'Все задания процесса ({len(done)} шт.) закрыты за секунды: {s}. Обычно проверка занимает ≈0.4 рабочих часа, руководитель ≈2 ч, бюджет ≈2.5 ч; '
            f'очень вероятно автоматическое или формальное «штамповое» согласование разными людьми.')
    # (и) аномально долгая проверка
    for _, r in g[(g.code=='C') & g.wh.notna()].iterrows():
        med = {'standard':0.40,'contract':0.41,'urgent':0.14}[rt]
        if r.wh >= 50:
            add(pid,'Аномально долгая проверка заявки',
                f'«Проверка заявки» (specialist) заняла {r.wh:.0f} рабочих часов ({r.sec/86400:.1f} календ. суток) при медиане {med} ч для {rt}; '
                f'обычный максимум других проверок ≈ 8 ч. Это отклонение по скорости (задание ждало в очереди специалиста), а не нарушение маршрута.')
        elif r.wh >= 7:
            add(pid,'Долгая проверка заявки (умеренно)',
                f'«Проверка заявки» заняла {r.wh:.1f} рабочих часа ({r.sec/3600:.0f} календ. ч) при медиане {med} ч для {rt} — в ≈20 раз выше нормы; '
                f'99-й перцентиль остальных проверок ≈ 1.3 ч. Возможно, обычное ожидание; кандидат на ручную проверку.')

A = pd.DataFrame(rows, columns=['ProcessInstanceId','AnomalyType','Explanation']).drop_duplicates()
A = A.sort_values(['AnomalyType','ProcessInstanceId'])
A.to_csv(OUT, index=False, encoding='utf-8')
print('\nстрок в anomalies.csv:', len(A), '| процессов:', A.ProcessInstanceId.nunique())
print(A.groupby('AnomalyType').ProcessInstanceId.nunique().sort_values(ascending=False))
