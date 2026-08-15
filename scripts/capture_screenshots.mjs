// Day 20 · Phase G — capture portfolio screenshots of the REAL running UI with headless Chrome.
//
// Needs the backend (CORS_ORIGINS=http://
// :5173) and `npm run dev` already running, plus
// puppeteer-core installed somewhere OUTSIDE the repo (no browser download — it drives installed Chrome):
//
//   mkdir /tmp/shots && cd /tmp/shots && npm init -y && PUPPETEER_SKIP_DOWNLOAD=1 npm i puppeteer-core
//   NODE_PATH=/tmp/shots/node_modules node scripts/capture_screenshots.mjs
//
// Drives the demo campaign A from docs/assets/campaign_A.json, writes PNGs to docs/assets/screenshots/,
// and prints any browser console errors / failed requests it saw (a free smoke test of the UI).

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const puppeteer = require("puppeteer-core");

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "docs/assets/screenshots");
const CAMPAIGN = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/assets/campaign_A.json"), "utf8"));
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const UI = process.env.UI_URL || "http://localhost:5173";
const API = process.env.API_URL || "http://localhost:8000";

const problems = [];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const toLocal = (s) => s.replace(" ", "T").slice(0, 16); // "2017-03-06 10:00:00" -> datetime-local

async function setValue(handle, value) {
  // React-controlled inputs ignore plain .value assignment; use the native setter + input event.
  await handle.evaluate((el, v) => {
    const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
}

async function fillForm(page) {
  await page.waitForSelector("form input", { timeout: 15000 }); // route change renders the form a tick later
  const fields = await page.$$("form input, form select");
  // order in CampaignForm: title, main category, sub-category, goal, currency, country, launch, deadline
  if (fields.length < 8) throw new Error(`expected 8 form controls, found ${fields.length}`);
  const values = [CAMPAIGN.name, CAMPAIGN.main_category, CAMPAIGN.category, String(CAMPAIGN.usd_goal_real),
    CAMPAIGN.currency, CAMPAIGN.country, toLocal(CAMPAIGN.launched), toLocal(CAMPAIGN.deadline)];
  for (let i = 0; i < 8; i++) await setValue(fields[i], values[i]);
}

async function submit(page) {
  const btn = await page.$("form button[type=submit]");
  if (!btn) throw new Error("no submit button found");
  await btn.click();
}

async function waitForText(page, text, timeout = 120000) {
  // textContent, not innerText: headings are CSS-uppercased, innerText would return "PREDICTION".
  await page.waitForFunction((t) => document.body.textContent.includes(t), { timeout }, text);
}

async function cardByHeading(page, text) {
  const h = await page.evaluateHandle((t) => {
    return [...document.querySelectorAll("h3,h4")].find((e) => e.textContent.trim().startsWith(t)) || null;
  }, text);
  const el = h.asElement();
  if (!el) return null;
  return el;
}

async function shotCard(page, headingText, file, ancestors = 1) {
  const h = await cardByHeading(page, headingText);
  if (!h) { console.log(`  ! heading not found: ${headingText}`); return; }
  const target = await h.evaluateHandle((e, n) => { let x = e; for (let i = 0; i < n; i++) x = x.parentElement; return x; }, ancestors);
  const el = target.asElement();
  await el.evaluate((e) => e.scrollIntoView({ block: "center" }));
  await sleep(250);
  await el.screenshot({ path: path.join(OUT, file) });
  console.log(`  saved ${file}`);
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "shell", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1.5 });
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) problems.push(`console.${m.type()}: ${m.text().slice(0, 200)}`); });
  page.on("pageerror", (e) => problems.push(`pageerror: ${String(e).slice(0, 200)}`));
  page.on("requestfailed", (r) => problems.push(`requestfailed: ${r.method()} ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => { if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.request().method()} ${r.url()}`); });

  // the sticky header would otherwise be captured on top of element screenshots taken mid-scroll;
  // evaluateOnNewDocument so it survives the full page loads (goto) between portals
  await page.evaluateOnNewDocument(() => {
    document.addEventListener("DOMContentLoaded", () => {
      const st = document.createElement("style");
      st.textContent = "header { position: static !important; }";
      document.head.appendChild(st);
    });
  });

  // 01 home
  await page.goto(UI, { waitUntil: "networkidle0" });
  await page.screenshot({ path: path.join(OUT, "01-home.png") });
  console.log("  saved 01-home.png");

  // sign in as founder (auth-ready stub)
  await page.goto(`${UI}/founder`, { waitUntil: "networkidle0" });
  if (page.url().includes("/login")) {
    await page.screenshot({ path: path.join(OUT, "00-login.png") });
    const [btn] = await page.$$("button");
    await btn.click();
    await page.waitForFunction(() => location.pathname === "/founder");
  }

  // Founder: submit campaign A
  await fillForm(page);
  await sleep(300);
  await submit(page);
  await waitForText(page, "Print / Save as PDF");
  await waitForText(page, "Similar campaigns that funded");
  await waitForText(page, "Priority improvements");
  await sleep(1500); // charts + scenario fetches settle
  await page.screenshot({ path: path.join(OUT, "02-founder-submitted.png") });
  console.log("  saved 02-founder-submitted.png");

  await shotCard(page, "Prediction", "03-prediction.png");
  await shotCard(page, "Why", "04-shap-bar-waterfall.png");
  await shotCard(page, "Historical evidence", "06-retrieval-tables.png");
  await shotCard(page, "Scores the platform defers", "07-deferred-scores.png");
  await shotCard(page, "Recommendations", "08-recommendations.png");
  await shotCard(page, "Founder Intelligence Report", "09-founder-report.png");

  // scenario simulator: drag the goal slider down to $10,000
  const slider = await page.$("input[type=range]");
  if (slider) {
    await setValue(slider, "10000");
    await sleep(2500); // debounce + /predict round trip
  }
  await shotCard(page, "Scenario simulator", "05-scenario-simulator.png");

  // full founder page (long) for reference
  await page.screenshot({ path: path.join(OUT, "02b-founder-full-page.png"), fullPage: true });
  console.log("  saved 02b-founder-full-page.png");

  // Investor
  await page.goto(`${UI}/investor`, { waitUntil: "networkidle0" });
  await fillForm(page);
  await sleep(300);
  await submit(page);
  await waitForText(page, "Overall confidence");
  await waitForText(page, "Print / Save as PDF");
  await sleep(1500);
  await shotCard(page, "Committee verdict", "10-investor-verdict.png");
  await shotCard(page, "Investor Due-Diligence Report", "10b-investor-report.png");
  await shotCard(page, "Risks & Red Flags", "11-investor-risks.png", 2);
  await shotCard(page, "Investment Decision", "12-investor-committee-votes.png", 2);

  // Mobile (founder)
  const m = await browser.newPage();
  await m.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true });
  await m.evaluateOnNewDocument(() => { document.addEventListener("DOMContentLoaded", () => { const st = document.createElement("style"); st.textContent = "header { position: static !important; }"; document.head.appendChild(st); }); });
  await m.goto(`${UI}/founder`, { waitUntil: "networkidle0" });
  await fillForm(m);
  await submit(m);
  await waitForText(m, "Priority improvements");
  await sleep(1500);
  await m.evaluate(() => window.scrollTo(0, 0));
  await sleep(300);
  await m.screenshot({ path: path.join(OUT, "15-mobile.png"), clip: { x: 0, y: 0, width: 390, height: 1700 } });
  console.log("  saved 15-mobile.png");
  // isMobile emulation widens the layout viewport to fit content, so compare against the real 390, not innerWidth.
  const overflow = await m.evaluate(() => document.documentElement.scrollWidth > 390);
  console.log(`  mobile horizontal overflow: ${overflow}`);

  // API docs (Swagger loads its assets from a CDN, so this needs internet)
  try {
    const d = await browser.newPage();
    await d.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 1.5 });
    await d.goto(`${API}/docs`, { waitUntil: "networkidle0", timeout: 30000 });
    await d.waitForSelector(".swagger-ui .opblock", { timeout: 15000 });
    await d.screenshot({ path: path.join(OUT, "13-api-docs.png") });
    console.log("  saved 13-api-docs.png");
  } catch (e) {
    console.log(`  ! API docs not captured (${String(e).slice(0, 80)})`);
  }

  await browser.close();
  console.log(problems.length ? `\nBrowser problems seen (${problems.length}):\n- ${[...new Set(problems)].join("\n- ")}` : "\nNo browser console errors or failed requests.");
}

main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
