const state = {
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
  interval: "1h",
  exchange: "bybit",
  loading: false,
  lastData: null,
  lastUpdatedAt: null,
};

const els = {
  refreshBtn: byId("refreshBtn"),
  intervalSelect: byId("intervalSelect"),
  statusDot: byId("statusDot"),
  statusValue: byId("statusValue"),
  errorBanner: byId("errorBanner"),
  modeLabel: byId("modeLabel"),
  headerSubtitle: byId("headerSubtitle"),
  equityValue: byId("equityValue"),
  equityMeta: byId("equityMeta"),
  cashValue: byId("cashValue"),
  positionCount: byId("positionCount"),
  positionMeta: byId("positionMeta"),
  pnlCard: byId("pnlCard"),
  pnlValue: byId("pnlValue"),
  pnlMeta: byId("pnlMeta"),
  lossValue: byId("lossValue"),
  riskCaps: byId("riskCaps"),
  assetCount: byId("assetCount"),
  autoRefresh: byId("autoRefresh"),
  signalGrid: byId("signalGrid"),
  edgeEventCount: byId("edgeEventCount"),
  edgeAvgScore: byId("edgeAvgScore"),
  edgeAllow: byId("edgeAllow"),
  edgeWeakAllow: byId("edgeWeakAllow"),
  edgeStrongAllow: byId("edgeStrongAllow"),
  edgeBlock: byId("edgeBlock"),
  edgeObserve: byId("edgeObserve"),
  edgeAvgEv: byId("edgeAvgEv"),
  edgeRecentList: byId("edgeRecentList"),
  recoveryUpdated: byId("recoveryUpdated"),
  recoveryReadyCount: byId("recoveryReadyCount"),
  recoveryReady: byId("recoveryReady"),
  recoveryOpen: byId("recoveryOpen"),
  recoveryClosed: byId("recoveryClosed"),
  recoveryWinRate: byId("recoveryWinRate"),
  recoveryAvg: byId("recoveryAvg"),
  recoveryPromotion: byId("recoveryPromotion"),
  recoveryPromotionReason: byId("recoveryPromotionReason"),
  recoveryOpenMeta: byId("recoveryOpenMeta"),
  recoveryReadyMeta: byId("recoveryReadyMeta"),
  recoveryClosedMeta: byId("recoveryClosedMeta"),
  recoveryOpenList: byId("recoveryOpenList"),
  recoveryReadyList: byId("recoveryReadyList"),
  recoveryClosedList: byId("recoveryClosedList"),
  paperUpdated: byId("paperUpdated"),
  paperMode: byId("paperMode"),
  paperOpen: byId("paperOpen"),
  paperClosed: byId("paperClosed"),
  paperWinRate: byId("paperWinRate"),
  paperAvg: byId("paperAvg"),
  paperRealized: byId("paperRealized"),
  paperUnrealized: byId("paperUnrealized"),
  paperDrawdown: byId("paperDrawdown"),
  paperOpenMeta: byId("paperOpenMeta"),
  paperLaneMeta: byId("paperLaneMeta"),
  paperClosedMeta: byId("paperClosedMeta"),
  paperOpenList: byId("paperOpenList"),
  paperLaneList: byId("paperLaneList"),
  paperClosedList: byId("paperClosedList"),
  dataSourcesList: byId("dataSourcesList"),
  positionsList: byId("positionsList"),
  exitList: byId("exitList"),
  weightsList: byId("weightsList"),
  journalList: byId("journalList"),
  monitorDaemon: byId("monitorDaemon"),
  monitorGenerated: byId("monitorGenerated"),
  monitorApproval: byId("monitorApproval"),
  monitorCounts: byId("monitorCounts"),
  monitorAccuracy: byId("monitorAccuracy"),
  monitorBacktest: byId("monitorBacktest"),
  monitorGap: byId("monitorGap"),
  monitorScenario: byId("monitorScenario"),
  monitorScenarioMeta: byId("monitorScenarioMeta"),
  monitorFlagCount: byId("monitorFlagCount"),
  monitorFlags: byId("monitorFlags"),
  monitorTopModels: byId("monitorTopModels"),
  monitorWeakModels: byId("monitorWeakModels"),
};

const usdFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const compactUsdFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 8,
});

els.refreshBtn.addEventListener("click", () => refresh({ manual: true }));
els.intervalSelect.addEventListener("change", (event) => {
  state.interval = event.target.value;
  refresh({ manual: true });
});

window.addEventListener("resize", debounce(() => {
  if (state.lastData) renderSignals(state.lastData.assets || []);
}, 150));

document.querySelectorAll(".nav-links a").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-links a").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});

renderSkeletons();
refresh();
setInterval(refresh, 30000);

async function refresh(options = {}) {
  if (state.loading) return;
  state.loading = true;
  setStatus("loading", options.manual ? "Refreshing" : "Syncing");
  els.refreshBtn.classList.add("is-loading");
  hideError();

  try {
    const params = new URLSearchParams({
      symbols: state.symbols.join(","),
      interval: state.interval,
      exchange: state.exchange,
    });
    const response = await fetch(`/api/snapshot?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);

    const data = await response.json();
    state.lastData = data;
    state.lastUpdatedAt = new Date();
    render(data);
    setStatus("online", "Online");
  } catch (error) {
    setStatus("error", "API error");
    showError(`${error.message}. Showing the last successful snapshot if available.`);
    if (!state.lastData) renderEmptyState();
  } finally {
    state.loading = false;
    els.refreshBtn.classList.remove("is-loading");
  }
}

function render(data) {
  const portfolio = data.portfolio || {};
  const config = data.config || {};
  const positions = getOpenPositions(portfolio);
  const unrealizedPnl = calculateTotalPnl(portfolio);
  const mode = config.bybit_demo ? "Demo mode" : config.live_trading_enabled ? "Live guarded" : "Paper mode";

  els.modeLabel.textContent = `${capitalize(data.exchange || state.exchange)} ${mode}`;
  els.headerSubtitle.textContent = `${data.interval || state.interval} market scan. Spot only, no leverage, no margin.`;
  els.equityValue.textContent = formatUsd(portfolio.equity || 0);
  els.equityMeta.textContent = `Last successful update ${formatTime(state.lastUpdatedAt)}`;
  els.cashValue.textContent = formatUsd(portfolio.cash || 0);
  els.positionCount.textContent = String(positions.length);
  els.positionMeta.textContent = positions.length ? `${positions.map((item) => item.symbol).join(", ")}` : "No exposure detected";
  els.pnlValue.textContent = formatUsd(unrealizedPnl);
  els.pnlMeta.textContent = unrealizedPnl >= 0 ? "Open exposure is positive" : "Open exposure is negative";
  els.pnlCard.classList.toggle("negative", unrealizedPnl < 0);
  els.pnlCard.classList.toggle("positive", unrealizedPnl >= 0);
  els.lossValue.textContent = String(portfolio.consecutive_losses || 0);
  els.riskCaps.textContent = `SL ${formatPct(config.stop_loss_pct)} / TP ${formatPct(config.take_profit_pct)} / Trail ${formatPct(config.trailing_stop_pct)}`;
  els.assetCount.textContent = `${(data.assets || []).length} symbols`;
  els.autoRefresh.textContent = `Last update ${formatTime(state.lastUpdatedAt)} | Auto-refresh 30s`;

  renderSignals(data.assets || [], portfolio);
  renderEdgeFilter(data.edge_filter_monitor || {});
  renderRecovery(data.recovery || {});
  renderPaperSimulator(data.paper_simulator || {});
  renderDataSources(data.data_sources || {});
  renderPositions(portfolio);
  renderExits(data.assets || []);
  renderLearning(data.learning || {});
  renderJournal(data.recent || []);
  renderLearningMonitor(data.learning_monitor || {});
}

function renderSignals(assets, portfolio = {}) {
  replaceChildren(els.signalGrid);

  if (!assets.length) {
    els.signalGrid.appendChild(emptyCard("No market signals yet", "Waiting for the next snapshot."));
    return;
  }

  assets.forEach((asset) => {
    const card = createEl("article", "signal-card");
    card.dataset.status = asset.status || "unknown";

    if (asset.status !== "ok") {
      append(card,
        createSignalHeader(asset.symbol, "ERROR", "sell"),
        textBlock("Data error", asset.error || "Unable to load this symbol.")
      );
      els.signalGrid.appendChild(card);
      return;
    }

    const decision = asset.decision || {};
    const edge = decision.edge_filter || {};
    const action = decision.action || "NO TRADE";
    const displayAction = displaySignalAction(asset.symbol, action, portfolio);
    const actionClass = actionClassName(action);
    const strategyLabel = displayStrategyLabel(asset.symbol, decision, portfolio);
    const probabilities = extractProbabilities(decision.market_state || "");
    const confidence = clamp(Number(decision.confidence || 0), 0, 100);
    const canvas = createEl("canvas", "sparkline");
    canvas.setAttribute("aria-label", `${asset.symbol} price sparkline`);

    const signalMeta = createEl("div", "signal-meta");
    append(signalMeta,
      statPill("Risk", decision.risk_level || "unknown", riskClass(decision.risk_level)),
      statPill("Strategy", strategyLabel, strategyLabel === "Active" ? "buy" : "neutral")
    );

    const details = createEl("details", "reason-details");
    const summary = createEl("summary");
    summary.textContent = trimText(decision.reasoning || "No reasoning supplied.", 155);
    const fullReason = createEl("p");
    fullReason.textContent = decision.reasoning || "No reasoning supplied.";
    const marketState = createEl("p", "market-state");
    marketState.textContent = decision.market_state || "Market state unavailable.";
    append(details, summary, fullReason, marketState);

    append(card,
      createSignalHeader(asset.symbol, displayAction, actionClass),
      createConfidenceRow(confidence),
      canvas,
      probabilityPanel(probabilities),
      orderBookPanel(asset.order_book),
      edgeFilterPanel(edge),
      signalMeta,
      details
    );

    els.signalGrid.appendChild(card);
    drawSparkline(canvas, asset.sparkline || [], actionClass);
  });
}

function displaySignalAction(symbol, action, portfolio) {
  if (action === "SELL" && !hasPortfolioPosition(symbol, portfolio)) return "SELL SIGNAL";
  return action;
}

function displayStrategyLabel(symbol, decision, portfolio) {
  const action = decision.action || "NO TRADE";
  const edge = decision.edge_filter || {};
  const edgeAction = String(edge.monitor_action || "").toUpperCase();
  if (action === "BUY" && ["ALLOW", "WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR", "RECOVERY_MONITOR"].includes(edgeAction)) {
    return shortStrategy(decision.selected_strategy || "Active");
  }
  if (action === "SELL" && hasPortfolioPosition(symbol, portfolio)) return "Exit Active";
  if (action === "SELL") return "Exit Signal";
  if (action === "BUY") return "Blocked Setup";
  return "Waiting";
}

function hasPortfolioPosition(symbol, portfolio) {
  const positions = portfolio.positions || {};
  return Number(positions[symbol] || 0) > 0;
}

function renderEdgeFilter(edgeMonitor) {
  const events = Number(edgeMonitor.events || 0);
  const daemon = edgeMonitor.daemon || {};
  els.edgeEventCount.textContent = `${events} events`;
  els.edgeAvgScore.textContent = daemon.state ? `Daemon ${daemon.state} | score ${Number(edgeMonitor.avg_score || 0).toFixed(1)}` : `Avg score ${Number(edgeMonitor.avg_score || 0).toFixed(1)}`;
  els.edgeAllow.textContent = String(edgeMonitor.allow || 0);
  els.edgeWeakAllow.textContent = String(edgeMonitor.weak_allow || 0);
  els.edgeStrongAllow.textContent = String(edgeMonitor.strong_allow || 0);
  els.edgeBlock.textContent = String(edgeMonitor.block || 0);
  els.edgeObserve.textContent = String(edgeMonitor.observe || 0);
  els.edgeAvgEv.textContent = `${Number(edgeMonitor.avg_ev_pct || 0).toFixed(4)}%`;

  replaceChildren(els.edgeRecentList);
  const recent = Array.isArray(edgeMonitor.recent) ? edgeMonitor.recent.slice(-6).reverse() : [];
  if (!recent.length) {
    els.edgeRecentList.appendChild(emptyCard("No edge events yet", "The monitor will populate after market snapshots."));
    return;
  }

  recent.forEach((event) => {
    const edge = event.edge_filter || {};
    const item = createEl("article", "edge-event");
    const head = createEl("div", "edge-event-head");
    append(head,
      createTextEl("strong", `${event.symbol || "Symbol"} ${event.interval || ""}`),
      badge(edge.monitor_action || "OBSERVE", edgeClass(edge.monitor_action))
    );
    append(item,
      head,
      row("Final action", event.final_action || "NO TRADE"),
      row("Score", `${Number(edge.score || 0).toFixed(0)}/100`),
      row("EV", edge.estimated_ev_pct === null || edge.estimated_ev_pct === undefined ? "n/a" : `${Number(edge.estimated_ev_pct).toFixed(4)}%`),
      createTextEl("p", trimText((edge.reasons || []).join(" "), 180))
    );
    els.edgeRecentList.appendChild(item);
  });
}

function renderPositions(portfolio) {
  replaceChildren(els.positionsList);
  const positions = getOpenPositions(portfolio);

  if (!positions.length) {
    els.positionsList.appendChild(emptyCard("No open positions", "The agent is currently holding cash."));
    return;
  }

  positions.forEach((position) => {
    const item = createEl("article", "position-card");
    const pnlClass = position.pnl >= 0 ? "profit" : "loss";
    append(item,
      row("Symbol", position.symbol, "strong"),
      row("Quantity", formatNumber(position.qty)),
      row("Current value", formatUsd(position.value)),
      row("Cost basis", formatUsd(position.cost)),
      row("Unrealized PnL", formatUsd(position.pnl), pnlClass)
    );
    els.positionsList.appendChild(item);
  });
}

function renderExits(assets) {
  replaceChildren(els.exitList);
  const validAssets = assets.filter((asset) => asset.status === "ok" && asset.exit);

  if (!validAssets.length) {
    els.exitList.appendChild(emptyCard("No exit data", "Exit checks will appear after a valid snapshot."));
    return;
  }

  validAssets.forEach((asset) => {
    const exit = asset.exit || {};
    const action = exit.action || "HOLD";
    const item = createEl("article", `exit-card ${action === "SELL" ? "urgent" : ""}`);
    const evidence = createEl("div", "evidence-list");
    const evidenceItems = Array.isArray(exit.evidence) && exit.evidence.length
      ? exit.evidence
      : ["No urgent exit evidence."];

    evidenceItems.slice(0, 4).forEach((entry) => {
      const line = createEl("p");
      line.textContent = entry;
      evidence.appendChild(line);
    });

    append(item,
      createSignalHeader(exit.symbol || asset.symbol, action, action === "SELL" ? "sell" : "wait"),
      row("Confidence", `${Number(exit.confidence || 0).toFixed(0)}%`),
      row("Unrealized PnL", formatUsd(exit.unrealized_pnl || 0), (exit.unrealized_pnl || 0) >= 0 ? "profit" : "loss"),
      evidence
    );
    els.exitList.appendChild(item);
  });
}

function renderLearning(learning) {
  replaceChildren(els.weightsList);
  const weights = learning.strategy_weights || {};
  const entries = Object.entries(weights);

  if (!entries.length) {
    els.weightsList.appendChild(emptyCard("No strategy weights yet", "Learning data will appear after journal activity."));
    return;
  }

  const maxWeight = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  entries.forEach(([name, weight]) => {
    const numeric = Number(weight) || 0;
    const pct = clamp((numeric / maxWeight) * 100, 4, 100);
    const item = createEl("article", "weight-card");
    const header = createEl("div", "weight-head");
    append(header, createTextEl("strong", formatStrategyName(name)), createTextEl("span", numeric.toFixed(2)));

    const track = createEl("div", "weight-track");
    const fill = createEl("span", "weight-fill");
    fill.style.width = `${pct}%`;
    track.appendChild(fill);
    append(item, header, track);
    els.weightsList.appendChild(item);
  });
}

function renderJournal(recent) {
  replaceChildren(els.journalList);
  const items = Array.isArray(recent) ? recent.slice(-6).reverse() : [];

  if (!items.length) {
    els.journalList.appendChild(emptyCard("No journal entries", "Recent agent decisions will be listed here."));
    return;
  }

  items.forEach((entry) => {
    const item = createEl("article", "timeline-item");
    const top = createEl("div", "timeline-top");
    const action = entry.action || entry.decision || "NO TRADE";
    append(top,
      createTextEl("strong", entry.symbol || "Symbol"),
      badge(action, actionClassName(action))
    );
    const body = createEl("p");
    body.textContent = trimText(entry.reasoning || entry.result || "Decision stored in journal.", 180);
    append(item, top, body);
    els.journalList.appendChild(item);
  });
}

function renderLearningMonitor(monitor) {
  const daemon = monitor.daemon || {};
  const health = monitor.model_health || {};
  const scenarioDistribution = health.current_scenario_distribution || {};
  const scenarioEntries = Object.entries(scenarioDistribution);
  const totalScenarioPredictions = scenarioEntries.reduce((sum, [, value]) => sum + Number(value || 0), 0);
  const dominantScenario = scenarioEntries.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
  const warnings = Array.isArray(monitor.risk_flags) ? monitor.risk_flags : [];

  els.monitorDaemon.textContent = `Daemon ${daemon.state || "unknown"}${daemon.cycle ? ` | cycle ${daemon.cycle}` : ""}`;
  els.monitorGenerated.textContent = monitor.generated_at ? `Generated ${formatDateTime(monitor.generated_at)}` : "Waiting for monitor";
  els.monitorApproval.textContent = `${formatPlainPct(health.approval_rate_pct)}%`;
  els.monitorCounts.textContent = `${health.approved || 0} approved / ${health.rejected || 0} rejected / ${health.errors || 0} errors`;
  els.monitorAccuracy.textContent = `${formatPlainPct(health.avg_accuracy)}%`;
  els.monitorBacktest.textContent = `${formatPlainPct(health.avg_backtest_accuracy)}%`;
  els.monitorGap.textContent = `Accuracy-backtest gap ${formatSigned(health.avg_accuracy_backtest_gap)} pts`;
  els.monitorScenario.textContent = dominantScenario ? capitalize(dominantScenario[0]) : "None";
  els.monitorScenarioMeta.textContent = dominantScenario
    ? `${dominantScenario[1]} of ${totalScenarioPredictions} current predictions`
    : "No current predictions";
  els.monitorFlagCount.textContent = `${warnings.length} flags`;

  renderMonitorFlags(warnings);
  renderMonitorModels(els.monitorTopModels, monitor.top_models || []);
  renderMonitorModels(els.monitorWeakModels, monitor.weak_models || []);
}

function renderMonitorFlags(flags) {
  replaceChildren(els.monitorFlags);
  if (!flags.length) {
    els.monitorFlags.appendChild(emptyCard("No active risk flags", "Training monitor does not see obvious health warnings yet."));
    return;
  }

  flags.forEach((flag) => {
    const levelClass = monitorLevelClass(flag.level);
    const item = createEl("article", `monitor-item ${levelClass}`);
    append(item, badge(String(flag.level || "info").toUpperCase(), levelClass), createTextEl("p", flag.message || "Monitor flag."));
    els.monitorFlags.appendChild(item);
  });
}

function renderMonitorModels(container, models) {
  replaceChildren(container);
  if (!models.length) {
    container.appendChild(emptyCard("No models yet", "The current training report has no model rows."));
    return;
  }

  models.slice(0, 6).forEach((model) => {
    const item = createEl("article", "monitor-model");
    const head = createEl("div", "monitor-model-head");
    append(head,
      createTextEl("strong", model.key || "MODEL"),
      badge(model.approved ? "APPROVED" : "WATCH", model.approved ? "buy" : "wait")
    );

    const meta = createEl("div", "monitor-model-grid");
    append(meta,
      metricMini("Accuracy", `${formatPlainPct(model.accuracy)}%`),
      metricMini("Backtest", `${formatPlainPct(model.backtest_accuracy)}%`),
      metricMini("Gap", formatSigned(model.accuracy_backtest_gap)),
      metricMini("Scenario", capitalize(model.current_scenario || "none"))
    );

    append(item, head, meta);
    if (Array.isArray(model.low_label_accuracy) && model.low_label_accuracy.length) {
      const warning = createEl("p", "model-warning");
      warning.textContent = `Weak class: ${model.low_label_accuracy.join(", ")}`;
      item.appendChild(warning);
    }
    container.appendChild(item);
  });
}

function metricMini(label, value) {
  const item = createEl("div", "metric-mini");
  append(item, createTextEl("span", label), createTextEl("strong", value));
  return item;
}

function renderSkeletons() {
  replaceChildren(els.signalGrid);
  for (let index = 0; index < 4; index += 1) {
    const skeleton = createEl("article", "signal-card skeleton-card");
    append(skeleton, createEl("div", "skeleton-line wide"), createEl("div", "skeleton-chart"), createEl("div", "skeleton-line"), createEl("div", "skeleton-line short"));
    els.signalGrid.appendChild(skeleton);
  }
}

function renderEmptyState() {
  replaceChildren(els.signalGrid);
  els.signalGrid.appendChild(emptyCard("Snapshot unavailable", "Check the dashboard server and try refresh again."));
}

function createSignalHeader(symbol, action, cls) {
  const header = createEl("div", "signal-header");
  append(header, createTextEl("strong", symbol || "UNKNOWN"), badge(action, cls));
  return header;
}

function createConfidenceRow(confidence) {
  const wrap = createEl("div", "confidence-row");
  const ring = createEl("div", "confidence-ring");
  ring.style.setProperty("--confidence", `${confidence}%`);
  ring.appendChild(createTextEl("span", `${confidence.toFixed(0)}%`));

  const copy = createEl("div");
  append(copy, createTextEl("span", "Confidence"), createTextEl("strong", confidenceLabel(confidence)));
  append(wrap, ring, copy);
  return wrap;
}

function probabilityPanel(probabilities) {
  const panel = createEl("div", "probability-panel");
  [
    ["Bullish", probabilities.bullish, "buy"],
    ["Bearish", probabilities.bearish, "sell"],
    ["Sideways", probabilities.sideways, "neutral"],
  ].forEach(([label, value, cls]) => panel.appendChild(probabilityBar(label, value, cls)));
  return panel;
}

function probabilityBar(label, value, cls) {
  const item = createEl("div", "probability-row");
  const labelEl = createTextEl("span", label);
  const track = createEl("span", "probability-track");
  const fill = createEl("span", `probability-fill ${cls}`);
  fill.style.width = `${clamp(value, 0, 100)}%`;
  track.appendChild(fill);
  append(item, labelEl, track, createTextEl("strong", `${Math.round(value)}%`));
  return item;
}

function orderBookPanel(orderBook) {
  const panel = createEl("div", "orderbook-panel");
  if (!orderBook || orderBook.status !== "ok") {
    append(panel, statPill("Depth", "Unavailable", "wait"));
    return panel;
  }

  const pressureClass = orderBook.pressure === "buy support" ? "buy" : orderBook.pressure === "sell wall" ? "sell" : "neutral";
  const qualityClass = orderBook.liquidity_quality === "strong" ? "buy" : orderBook.liquidity_quality === "weak" ? "sell" : "wait";
  append(panel,
    statPill("Spread", `${Number(orderBook.spread_pct || 0).toFixed(4)}%`, qualityClass),
    statPill("Pressure", orderBook.pressure || "balanced", pressureClass),
    statPill("Depth", orderBook.liquidity_quality || "unknown", qualityClass),
    statPill("Slip $25", `${Number(orderBook.estimated_slippage_pct_25_usdt || 0).toFixed(4)}%`, qualityClass)
  );
  return panel;
}

function edgeFilterPanel(edge) {
  const panel = createEl("div", "edge-panel");
  if (!edge || !edge.mode) {
    append(panel, statPill("Edge", "Pending", "wait"));
    return panel;
  }
  append(panel,
    statPill("Edge", edge.monitor_action || "OBSERVE", edgeClass(edge.monitor_action)),
    statPill("Score", `${Number(edge.score || 0).toFixed(0)}/100`, Number(edge.score || 0) >= 55 ? "buy" : "wait"),
    statPill("EV", edge.estimated_ev_pct === null || edge.estimated_ev_pct === undefined ? "n/a" : `${Number(edge.estimated_ev_pct).toFixed(4)}%`, Number(edge.estimated_ev_pct || 0) > 0 ? "buy" : "sell"),
    statPill("Regime", edge.regime || "unknown", "neutral")
  );
  const reason = createTextEl("p", trimText((edge.reasons || []).join(" "), 120), "edge-reason");
  panel.appendChild(reason);
  return panel;
}

function renderRecovery(recovery) {
  const watch = recovery.watch || {};
  const shadow = recovery.shadow || {};
  const summary = shadow.summary || {};
  const promotion = shadow.promotion || {};
  const ready = Array.isArray(watch.ready_now) ? watch.ready_now : [];
  const openTrades = Array.isArray(shadow.open_trades) ? shadow.open_trades : [];
  const closedTrades = Array.isArray(shadow.recent_closed) ? shadow.recent_closed : [];

  els.recoveryUpdated.textContent = shadow.generated_at ? `Updated ${formatDateTime(shadow.generated_at)}` : "Waiting for shadow report";
  els.recoveryReadyCount.textContent = `${ready.length} ready`;
  els.recoveryReady.textContent = String(ready.length);
  els.recoveryOpen.textContent = String(summary.open || openTrades.length || 0);
  els.recoveryClosed.textContent = String(summary.closed || closedTrades.length || 0);
  els.recoveryWinRate.textContent = `${formatPlainPct(summary.win_rate_pct)}%`;
  els.recoveryAvg.textContent = `Avg ${formatSigned(summary.avg_realized_pct)}% | Open ${formatSigned(summary.avg_unrealized_pct)}%`;
  els.recoveryPromotion.textContent = promotion.status === "ELIGIBLE_FOR_PAPER_ALLOW_SMALL" ? "Eligible" : "Not eligible";
  els.recoveryPromotionReason.textContent = promotion.reason || "Waiting for shadow evidence";
  els.recoveryOpenMeta.textContent = `${openTrades.length} open`;
  els.recoveryReadyMeta.textContent = `${ready.length} ready`;
  els.recoveryClosedMeta.textContent = `${closedTrades.length} closed`;

  renderRecoveryTrades(els.recoveryOpenList, openTrades, "No open shadow trades", "Recovery Monitor has not opened a shadow trade.");
  renderRecoveryReady(ready);
  renderRecoveryTrades(els.recoveryClosedList, closedTrades.slice(-6).reverse(), "No closed shadow trades", "TP/SL outcomes will appear here.");
}

function renderRecoveryReady(items) {
  replaceChildren(els.recoveryReadyList);
  if (!items.length) {
    els.recoveryReadyList.appendChild(emptyCard("No ready recovery pockets", "Current market does not match a recovery setup."));
    return;
  }
  items.slice(0, 6).forEach((item) => {
    const card = createEl("article", "monitor-model");
    const head = createEl("div", "monitor-model-head");
    append(head, createTextEl("strong", `${item.symbol || "Symbol"} ${item.interval || ""}`), badge(item.edge_action || "RECOVERY", "wait"));
    const meta = createEl("div", "monitor-model-grid");
    append(meta,
      metricMini("Strategy", shortStrategy(item.strategy)),
      metricMini("Entry", formatNumber(item.price)),
      metricMini("RSI", Number(item.rsi || 0).toFixed(2)),
      metricMini("Audit Avg", `${formatSigned((item.audit || {}).avg_realized_pct)}%`)
    );
    append(card, head, meta);
    els.recoveryReadyList.appendChild(card);
  });
}

function renderRecoveryTrades(container, trades, emptyTitle, emptyBody) {
  replaceChildren(container);
  if (!trades.length) {
    container.appendChild(emptyCard(emptyTitle, emptyBody));
    return;
  }
  trades.slice(0, 6).forEach((trade) => {
    const card = createEl("article", "monitor-model");
    const head = createEl("div", "monitor-model-head");
    const outcome = trade.outcome || trade.status || "open";
    append(head, createTextEl("strong", `${trade.symbol || "Symbol"} ${trade.interval || ""}`), badge(String(outcome).toUpperCase(), recoveryOutcomeClass(outcome)));
    const meta = createEl("div", "monitor-model-grid");
    append(meta,
      metricMini("Entry", formatNumber(trade.entry_price)),
      metricMini("Progress", `${trade.future_candles || 0}/${trade.required_candles || "-"}`),
      metricMini("PnL", trade.realized_pct === undefined ? `${formatSigned(trade.unrealized_pct)}%` : `${formatSigned(trade.realized_pct)}%`),
      metricMini("TP / SL", trade.distance_to_tp_pct === undefined ? `${formatPlainPct((trade.audit || {}).win_rate_pct)}% WR` : `${formatSigned(trade.distance_to_tp_pct)} / ${formatSigned(trade.distance_to_sl_pct)}%`)
    );
    append(card, head, meta);
    container.appendChild(card);
  });
}

function renderPaperSimulator(paper) {
  const summary = paper.summary || {};
  const openTrades = Array.isArray(paper.open_trades) ? paper.open_trades : [];
  const closedTrades = Array.isArray(paper.recent_closed) ? paper.recent_closed : [];
  const lanes = Array.isArray(paper.by_lane) ? paper.by_lane : [];

  els.paperUpdated.textContent = paper.generated_at ? `Updated ${formatDateTime(paper.generated_at)}` : "Waiting for paper report";
  els.paperMode.textContent = paper.execution_enabled === false ? "Monitor only" : "Unknown mode";
  els.paperOpen.textContent = String(summary.open || openTrades.length || 0);
  els.paperClosed.textContent = String(summary.closed || closedTrades.length || 0);
  els.paperWinRate.textContent = `${formatPlainPct(summary.win_rate_pct)}%`;
  els.paperAvg.textContent = `Avg ${formatSigned(summary.avg_realized_pct)}%`;
  els.paperRealized.textContent = formatUsd(summary.realized_quote || 0);
  els.paperUnrealized.textContent = `Open ${formatUsd(summary.unrealized_quote || 0)}`;
  els.paperDrawdown.textContent = `${formatSigned(summary.max_drawdown_pct)}%`;
  els.paperOpenMeta.textContent = `${openTrades.length} open`;
  els.paperLaneMeta.textContent = `${lanes.length} lanes`;
  els.paperClosedMeta.textContent = `${closedTrades.length} closed`;

  renderPaperTrades(els.paperOpenList, openTrades, "No open paper trades", "The simulator has not opened a virtual trade.");
  renderPaperLanes(lanes);
  renderPaperTrades(els.paperClosedList, closedTrades.slice(-6).reverse(), "No closed paper trades", "Closed TP/SL results will appear here.");
}

function renderPaperLanes(lanes) {
  replaceChildren(els.paperLaneList);
  if (!lanes.length) {
    els.paperLaneList.appendChild(emptyCard("No paper lanes", "Waiting for simulated trades."));
    return;
  }
  lanes.forEach((lane) => {
    const card = createEl("article", "monitor-model");
    const head = createEl("div", "monitor-model-head");
    append(head, createTextEl("strong", lane.lane || "Lane"), badge(`${formatPlainPct(lane.win_rate_pct)}% WR`, Number(lane.avg_realized_pct || 0) > 0 ? "buy" : "wait"));
    const meta = createEl("div", "monitor-model-grid");
    append(meta,
      metricMini("Open", lane.open || 0),
      metricMini("Closed", lane.closed || 0),
      metricMini("Avg", `${formatSigned(lane.avg_realized_pct)}%`),
      metricMini("PnL", formatUsd(lane.realized_quote || 0))
    );
    append(card, head, meta);
    els.paperLaneList.appendChild(card);
  });
}

function renderPaperTrades(container, trades, emptyTitle, emptyBody) {
  replaceChildren(container);
  if (!trades.length) {
    container.appendChild(emptyCard(emptyTitle, emptyBody));
    return;
  }
  trades.slice(0, 6).forEach((trade) => {
    const card = createEl("article", "monitor-model");
    const head = createEl("div", "monitor-model-head");
    const outcome = trade.outcome || trade.status || "open";
    append(head, createTextEl("strong", `${trade.symbol || "Symbol"} ${trade.interval || ""}`), badge(String(outcome).toUpperCase(), recoveryOutcomeClass(outcome)));
    const meta = createEl("div", "monitor-model-grid");
    append(meta,
      metricMini("Lane", shortLane(trade.lane)),
      metricMini("Entry", formatNumber(trade.entry_price)),
      metricMini("Score", trade.score === undefined || trade.score === null ? "-" : Number(trade.score).toFixed(0)),
      metricMini("PnL", trade.realized_pct === undefined ? `${formatSigned(trade.unrealized_pct)}%` : `${formatSigned(trade.realized_pct)}%`)
    );
    append(card, head, meta);
    container.appendChild(card);
  });
}

function renderDataSources(dataSources) {
  replaceChildren(els.dataSourcesList);
  const groups = [
    ...(dataSources.historical || []),
    ...(dataSources.microstructure || []),
  ];
  if (!groups.length) {
    els.dataSourcesList.appendChild(emptyCard("No source metadata", "The dashboard has not received source status yet."));
    return;
  }

  groups.forEach((source) => {
    const card = createEl("article", "source-card");
    append(card,
      createSignalHeader(source.name || "Source", String(source.status || "unknown").toUpperCase(), source.status === "active" || source.status === "available" ? "buy" : "wait"),
      createTextEl("p", source.usage || "Market data source.")
    );
    els.dataSourcesList.appendChild(card);
  });
}

function statPill(label, value, cls) {
  const pill = createEl("div", `stat-pill ${cls}`);
  append(pill, createTextEl("span", label), createTextEl("strong", value));
  return pill;
}

function emptyCard(title, body) {
  const item = createEl("article", "empty-card");
  append(item, createTextEl("strong", title), createTextEl("p", body));
  return item;
}

function textBlock(title, body) {
  const block = createEl("div", "text-block");
  append(block, createTextEl("strong", title), createTextEl("p", body));
  return block;
}

function row(label, value, valueClass = "") {
  const line = createEl("div", "data-row");
  append(line, createTextEl("span", label), createTextEl(valueClass === "strong" ? "strong" : "b", value, valueClass));
  return line;
}

function badge(text, cls) {
  const item = createEl("span", `action-badge ${cls}`);
  item.textContent = text;
  return item;
}

function drawSparkline(canvas, values, cls) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * scale);
  canvas.height = Math.floor(height * scale);

  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cleanValues = (values || []).map(Number).filter((value) => Number.isFinite(value));
  if (cleanValues.length < 2) return;

  const min = Math.min(...cleanValues);
  const max = Math.max(...cleanValues);
  const span = max - min || 1;
  const color = cls === "buy" ? "#2be39b" : cls === "sell" ? "#ff5d73" : "#f4b740";
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, `${hexToRgba(color, 0.24)}`);
  gradient.addColorStop(1, `${hexToRgba(color, 0)}`);

  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.beginPath();

  const points = cleanValues.map((value, index) => ({
    x: (index / (cleanValues.length - 1)) * width,
    y: height - ((value - min) / span) * (height - 18) - 9,
  }));

  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  ctx.lineTo(width, height);
  ctx.lineTo(0, height);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
}

function getOpenPositions(portfolio) {
  const positions = portfolio.positions || {};
  return Object.entries(positions)
    .filter(([, qty]) => Number(qty) > 0)
    .map(([symbol, qty]) => {
      const numericQty = Number(qty) || 0;
      const price = Number((portfolio.last_prices || {})[symbol]) || 0;
      const cost = Number((portfolio.cost_basis || {})[symbol]) || 0;
      const value = numericQty * price;
      return { symbol, qty: numericQty, price, cost, value, pnl: value - cost };
    });
}

function calculateTotalPnl(portfolio) {
  return getOpenPositions(portfolio).reduce((sum, position) => sum + position.pnl, 0);
}

function extractProbabilities(text) {
  return {
    bullish: extractProbability(text, "bullish"),
    bearish: extractProbability(text, "bearish"),
    sideways: extractProbability(text, "sideways"),
  };
}

function extractProbability(text, key) {
  const patterns = [
    new RegExp(`${key}\\s*[:=]?\\s*(\\d+(?:\\.\\d+)?)%`, "i"),
    new RegExp(`${key}[^0-9]{0,16}(\\d+(?:\\.\\d+)?)`, "i"),
  ];
  for (const pattern of patterns) {
    const match = String(text || "").match(pattern);
    if (match) return clamp(Number(match[1]) || 0, 0, 100);
  }
  return 0;
}

function setStatus(type, label) {
  els.statusValue.textContent = label;
  els.statusDot.className = `status-dot ${type}`;
}

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.hidden = false;
}

function hideError() {
  els.errorBanner.hidden = true;
  els.errorBanner.textContent = "";
}

function createEl(tag, className = "") {
  const el = document.createElement(tag);
  if (className) el.className = className;
  return el;
}

function createTextEl(tag, text, className = "") {
  const el = createEl(tag, className);
  el.textContent = text;
  return el;
}

function append(parent, ...children) {
  children.forEach((child) => parent.appendChild(child));
  return parent;
}

function replaceChildren(parent) {
  while (parent.firstChild) parent.removeChild(parent.firstChild);
}

function byId(id) {
  return document.getElementById(id);
}

function actionClassName(action) {
  if (action === "BUY") return "buy";
  if (action === "SELL") return "sell";
  return "wait";
}

function riskClass(risk) {
  const normalized = String(risk || "").toLowerCase();
  if (normalized.includes("high")) return "sell";
  if (normalized.includes("medium")) return "wait";
  if (normalized.includes("low")) return "buy";
  return "neutral";
}

function monitorLevelClass(level) {
  const normalized = String(level || "").toLowerCase();
  if (normalized === "high") return "sell";
  if (normalized === "medium") return "wait";
  return "neutral";
}

function edgeClass(action) {
  const normalized = String(action || "").toUpperCase();
  if (["ALLOW", "WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"].includes(normalized)) return "buy";
  if (normalized === "RECOVERY_MONITOR") return "wait";
  if (normalized === "BLOCK") return "sell";
  return "wait";
}

function recoveryOutcomeClass(outcome) {
  const normalized = String(outcome || "").toLowerCase();
  if (normalized.includes("take_profit")) return "buy";
  if (normalized.includes("stop_loss")) return "sell";
  return "wait";
}

function confidenceLabel(value) {
  if (value >= 75) return "Strong";
  if (value >= 55) return "Moderate";
  if (value >= 35) return "Cautious";
  return "Low signal";
}

function formatUsd(value) {
  const numeric = Number(value) || 0;
  if (Math.abs(numeric) >= 100000) return compactUsdFormatter.format(numeric);
  return usdFormatter.format(numeric);
}

function formatNumber(value) {
  return numberFormatter.format(Number(value) || 0);
}

function formatPct(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

function formatTime(date) {
  if (!date) return "pending";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function formatPlainPct(value) {
  return Number(value || 0).toFixed(2).replace(/\.00$/, "");
}

function formatSigned(value) {
  const numeric = Number(value || 0);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(2)}`;
}

function formatStrategyName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function shortLane(lane) {
  return String(lane || "Paper").replace(/_PAPER$/, "").replace(/_/g, " ");
}

function shortStrategy(name) {
  return String(name || "")
    .replace(" Strategy", "")
    .replace("Mean Reversion", "Mean Rev.")
    .replace("Trend Following", "Trend")
    .replace("Breakout", "Breakout");
}

function trimText(text, limit) {
  const clean = String(text || "").trim();
  if (clean.length <= limit) return clean;
  return `${clean.slice(0, limit - 1)}...`;
}

function capitalize(text) {
  const value = String(text || "");
  return value ? `${value[0].toUpperCase()}${value.slice(1)}` : "";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function debounce(fn, wait) {
  let timeout;
  return (...args) => {
    window.clearTimeout(timeout);
    timeout = window.setTimeout(() => fn(...args), wait);
  };
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const red = parseInt(value.slice(0, 2), 16);
  const green = parseInt(value.slice(2, 4), 16);
  const blue = parseInt(value.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}
