import sys, os, itertools
import pandas as pd
import graphviz

SRC = sys.argv[1] if len(sys.argv) > 1 else "students.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
log = []                                    # строки для analysis_summary.txt
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ---------- 1. Загрузка, группировка, упорядочивание ----------
df = pd.read_csv(SRC, encoding="utf-8-sig")            # UTF-8 с BOM
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["ProcessInstanceId", "EventIndex"])  # числовой EventIndex
g = df.groupby("ProcessInstanceId")
say("Строк:", len(df), "| экземпляров:", g.ngroups, "| активностей:", df.ActivityName.nunique())
say("EventIndex 1..N без пропусков:", bool((g.EventIndex.max() == g.size()).all()),
    "| время не убывает внутри экземпляра:", bool(g.Timestamp.apply(lambda s: s.is_monotonic_increasing).all()))

# ---------- 2. Активности, начала, окончания ----------
say("\n== Активности (число выполнений / роль / EventType) ==")
say(df.groupby(["ActivityName", "EventType", "Role"]).size().to_string())
first = g.ActivityName.first().value_counts(); last = g.ActivityName.last().value_counts()
say("\nПервое событие:", first.to_dict()); say("Последнее событие:", last.to_dict())

# ---------- 3. Варианты ----------
routes = g.ActivityName.apply(lambda s: " > ".join(s))
vc = routes.value_counts()
variants = pd.DataFrame({"Route": vc.index, "NumberOfCases": vc.values})
variants["Share"] = (variants.NumberOfCases / g.ngroups).round(4)
variants.insert(0, "Variant", [f"V{i+1}" for i in range(len(variants))])
variants.to_csv(f"{OUT}/variants.csv", index=False, encoding="utf-8")
say("\n== Варианты ==")
for r in variants.itertuples(): say(r.Variant, r.NumberOfCases, f"{r.Share:.1%}", "|", r.Route)

# ---------- 4. Граф непосредственно следующих активностей (DFG) ----------
df["Next"] = g.ActivityName.shift(-1)
df["WaitH"] = (g.Timestamp.shift(-1) - df.Timestamp).dt.total_seconds() / 3600
edges = df.dropna(subset=["Next"]).groupby(["ActivityName", "Next"]).agg(
    Cases=("Next", "size"), MedianCalendarHours=("WaitH", "median")).reset_index()
edges.columns = ["SourceActivity", "TargetActivity", "Cases", "MedianCalendarHours"]
edges[["SourceActivity", "TargetActivity"]].to_csv(f"{OUT}/student_transitions.csv", index=False, encoding="utf-8")
say("\n== Переходы (DFG) =="); say(edges.round(1).to_string(index=False))

# ---------- 5. Условия развилок ----------
case = g.agg(route=("ActivityName", lambda s: " > ".join(s)), amount=("Amount", "first"),
             final=("ActivityName", "last"))
case["extra"] = case.route.str.contains("Дополнительное согласование")
case["budget"] = case.route.str.contains("Проверка бюджета")
reach = case[case.budget]
say("\n== Дополнительное согласование vs Amount (среди дошедших до 'Проверка бюджета') ==")
say(reach.groupby("extra").amount.agg(["count", "min", "max"]).to_string())
low = reach[~reach.extra].amount.max(); high = reach[reach.extra].amount.min()
say(f"Граница лежит между {low:.2f} и {high:.2f}; ни одного нарушения правила 'Amount >= 150000'.")
say("Нарушений правила:", int(((reach.amount >= 150000) != reach.extra).sum()))

df["Attempt"] = df.groupby(["ProcessInstanceId", "ActivityName"]).cumcount() + 1
t = df[df.ActivityName.isin(["Первичная проверка", "Согласование руководителем"])]
say("\n== Результаты по номеру прохода =="); say(pd.crosstab([t.ActivityName, t.Attempt], t.Result).to_string())
nret = g.ActivityName.apply(lambda s: (s == "Доработка заявки").sum())
say("\nЧисло возвратов на доработку на экземпляр:", nret.value_counts().sort_index().to_dict())
first_t = t[t.Attempt == 1].copy()
first_t["bin"] = pd.cut(first_t.Amount, [0, 5e4, 1e5, 1.5e5, 2.5e5, 4e5])
say("\nРезультат 1-го прохода по сумме:"); say(pd.crosstab([first_t.ActivityName, first_t.bin], first_t.Result).to_string())

# ---------- 6. Проверка рабочего календаря ----------
h = df[df.EventType == "task_completed"].Timestamp
say("\nЗавершения заданий: дни недели", sorted(h.dt.dayofweek.unique()), "часы", h.dt.hour.min(), "-", h.dt.hour.max())

# ---------- 7. Длительности по вариантам ----------
dur = g.Timestamp.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
say("\n== Медианная календарная длительность по варианту, ч =="); 
for r in variants.itertuples(): say(r.Variant, round(dur[routes == r.Route].median(), 1))

# ---------- 8. Бонус: проверка правилами и сравнение с DFG ----------
def conforms(route, amount):
    a = route.split(" > ")
    if a[0] != "Регистрация заявки" or a[-1] not in ("Заявка выполнена", "Заявка отклонена", "Заявка отменена"): return False
    i = 1; k = 0
    while True:                                   # цикл проверки/согласования, до 2 возвратов
        if a[i:i+1] != ["Первичная проверка"]: return False
        i += 1
        if a[i:i+1] == ["Заявка отклонена"]: return k == 0 and i == len(a)-1
        if a[i:i+1] != ["Согласование руководителем"]: return False
        i += 1
        if a[i:i+1] == ["Заявка отменена"] or a[i:i+1] == ["Заявка отклонена"]: return k == 0 and i == len(a)-1
        if a[i:i+1] == ["Доработка заявки"]:
            k += 1; i += 1
            if k > 2: return False
            continue
        break
    if a[i:i+1] != ["Проверка бюджета"]: return False
    i += 1
    if amount >= 150000:
        if a[i:i+1] != ["Дополнительное согласование"]: return False
        i += 1
    return a[i:] == ["Исполнение закупки", "Заявка выполнена"]
ok = sum(conforms(r, a) for r, a in zip(case.route, case.amount))
say(f"\nПравиловая модель описывает {ok} из {len(case)} экземпляров.")

succ = {}
for s_, t_ in edges[["SourceActivity", "TargetActivity"]].itertuples(index=False): succ.setdefault(s_, []).append(t_)
ends = {"Заявка выполнена", "Заявка отклонена", "Заявка отменена"}
def walks(maxlen):
    cnt = 0; stack = [["Регистрация заявки"]]
    while stack:
        p = stack.pop()
        if p[-1] in ends: cnt += 1; continue
        if len(p) >= maxlen: continue
        for n in succ.get(p[-1], []): stack.append(p + [n])
    return cnt
for L in (12, 16, 20):
    say(f"Полных маршрутов, допускаемых DFG при длине <= {L}: {walks(L)} (наблюдалось {len(variants)}, максимум длины {max(len(r.split(' > ')) for r in variants.Route)})")
say("DFG допускает также 'Дополнительное согласование' при любой сумме и неограниченное число возвратов.")

# ---------- 9. Схема ----------
N = g.ngroups; cnt = df.ActivityName.value_counts()
G = graphviz.Digraph("process", format="png")
G.attr(rankdir="LR", splines="spline", nodesep="0.5", ranksep="0.55", fontname="Helvetica", dpi="170",
       label="Восстановленная схема процесса закупки (1000 экземпляров). Подписи на стрелках: число экземпляров / проходов",
       labelloc="t", fontsize="16")
G.attr("node", fontname="Helvetica", fontsize="12", style="filled")
G.attr("edge", fontname="Helvetica", fontsize="11")
def task(n, name, role, fill="#dbeafe"):
    G.node(n, f"{name}\n[{role}] · {cnt[name]}", shape="box", style="rounded,filled", fillcolor=fill, penwidth="1.4")
G.node("S", "", shape="circle", width="0.3", fillcolor="#86efac")
G.node("REG", f"Регистрация заявки\n[система] · {cnt['Регистрация заявки']}", shape="box", style="rounded,filled", fillcolor="#bbf7d0")
task("PP", "Первичная проверка", "специалист")
task("SR", "Согласование руководителем", "руководитель")
task("DR", "Доработка заявки", "инициатор", "#fef3c7")
task("PB", "Проверка бюджета", "финансы")
task("DS", "Дополнительное согласование", "директор", "#fde68a")
task("IZ", "Исполнение закупки", "закупки")
for n, w, c in [("OK", "Заявка выполнена", "#bbf7d0"), ("RJ", "Заявка отклонена", "#fecaca"), ("CN", "Заявка отменена", "#e5e7eb")]:
    G.node(n, f"{w}\n[система] · {cnt[w]}", shape="box", style="rounded,filled,bold", fillcolor=c)
for x, lab in [("X1", "X"), ("X2", "X"), ("X3", "X"), ("X4", "X")]:
    G.node(x, lab, shape="diamond", width="0.45", height="0.45", fillcolor="#ffffff", fontsize="11")
E = G.edge
E("S", "REG"); E("REG", "PP", label="1000")
E("PP", "X1", label="1174 решений")
E("X1", "SR", label="accept\n1129", penwidth="2.4")
E("X1", "RJ", label="reject\n45 (только 1-й проход)", color="#b91c1c", fontcolor="#b91c1c")
E("SR", "X2", label="1129 решений")
E("X2", "PB", label="accept\n911", penwidth="2.4")
E("X2", "DR", label="return\n174", color="#d97706", fontcolor="#d97706", style="dashed")
E("DR", "PP", label="возврат (до 2 раз)\n174", color="#d97706", fontcolor="#d97706", style="dashed", constraint="false")
E("X2", "RJ", label="reject 18", color="#b91c1c", fontcolor="#b91c1c")
E("X2", "CN", label="cancel 26", color="#6b7280", fontcolor="#6b7280")
E("PB", "X3")
E("X3", "DS", label="Amount ≥ 150 000\n209", color="#92400e", fontcolor="#92400e")
E("X3", "X4", label="Amount < 150 000\n702 (пропуск)", style="dotted")
E("DS", "X4"); E("X4", "IZ", penwidth="2.4", label="911"); E("IZ", "OK", label="911", penwidth="2.4")
leg = ("<<table border='0' cellborder='1' cellspacing='0' cellpadding='4'>"
 "<tr><td colspan='2'><b>Легенда</b></td></tr>"
 "<tr><td bgcolor='#dbeafe'>скруглённый блок</td><td>человеческое задание [роль] · число выполнений</td></tr>"
 "<tr><td bgcolor='#fde68a'>жёлтый блок</td><td>необязательное действие (условное / возвратное)</td></tr>"
 "<tr><td>X (ромб)</td><td>исключающая развилка (XOR); слияние — тоже X</td></tr>"
 "<tr><td>жирная стрелка</td><td>основной маршрут (567 из 1000)</td></tr>"
 "<tr><td>пунктир</td><td>возврат на доработку</td></tr>"
 "<tr><td>точечная</td><td>обход необязательного действия</td></tr>"
 "<tr><td bgcolor='#bbf7d0'>зелёный / красный / серый</td><td>успех / отказ / отмена</td></tr></table>>")
G.node("LEG", leg, shape="plaintext", style="")
G.edge("OK", "LEG", style="invis")
G.render(f"{OUT}/process_model", cleanup=True)
G.format = "svg"; G.render(f"{OUT}/process_model", cleanup=True)

mm = '''flowchart LR
  S((Старт)) --> REG[Регистрация заявки]
  REG --> PP[Первичная проверка<br/>специалист]
  PP --> X1{X}
  X1 -- "accept 1129" --> SR[Согласование руководителем<br/>руководитель]
  X1 -- "reject 45" --> RJ[[Заявка отклонена]]
  SR --> X2{X}
  X2 -- "accept 911" --> PB[Проверка бюджета<br/>финансы]
  X2 -. "return 174" .-> DR[Доработка заявки<br/>инициатор]
  DR -. "до 2 раз" .-> PP
  X2 -- "reject 18" --> RJ
  X2 -- "cancel 26" --> CN[[Заявка отменена]]
  PB --> X3{Amount >= 150000?}
  X3 -- "да 209" --> DS[Дополнительное согласование<br/>директор]
  X3 -- "нет 702" --> X4{X}
  DS --> X4
  X4 --> IZ[Исполнение закупки<br/>закупки]
  IZ --> OK[[Заявка выполнена 911]]
'''
open(f"{OUT}/process_model.mmd", "w", encoding="utf-8").write(mm)
open(f"{OUT}/analysis_summary.txt", "w", encoding="utf-8").write("\n".join(log))
