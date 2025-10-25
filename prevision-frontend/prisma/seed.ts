import { PrismaClient } from "@prisma/client";
import fs from "fs";
import path from "path";

const prisma = new PrismaClient();

function fileExists(p: string) {
  try {
    fs.accessSync(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function seedTemperature() {
  const possiblePaths = [
    path.resolve(__dirname, "../..", "data_temperature_sceaux.json"),
    path.resolve(
      __dirname,
      "../../prevision-python",
      "data_temperature_sceaux.json"
    ),
  ];
  const found = possiblePaths.find(fileExists);
  if (!found) {
    console.warn(
      "[seed] Arquivo data_temperature_sceaux.json não encontrado. Pulando temperaturas."
    );
    return;
  }
  console.log(`[seed] Importando temperaturas de: ${found}`);
  const raw = await fs.promises.readFile(found, "utf-8");
  const data = JSON.parse(raw);
  const times: string[] = data?.hourly?.time ?? [];
  const temps: number[] = data?.hourly?.temperature_2m ?? [];
  if (!times.length || !temps.length) {
    console.warn(
      "[seed] JSON de temperatura não possui campos esperados (hourly.time, hourly.temperature_2m)"
    );
    return;
  }
  const limit = Number(process.env.SEED_TEMP_LIMIT ?? 5000);
  const rows = times.slice(0, limit).map((t, i) => ({
    timestamp: new Date(t),
    temp: Number(temps[i]),
    location: "Sceaux",
  }));

  // Batch insert
  const batchSize = 1000;
  for (let i = 0; i < rows.length; i += batchSize) {
    const batch = rows.slice(i, i + batchSize);
    await prisma.temperatureReading.createMany({
      data: batch,
      skipDuplicates: true,
    });
    console.log(
      `[seed] Temperaturas inseridas: ${i + batch.length}/${rows.length}`
    );
  }
}

function parseNumber(value: string | undefined): number | null {
  if (value == null) return null;
  const v = value.trim();
  if (v === "?" || v === "") return null;
  const n = Number(v.replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

async function seedPower() {
  const possiblePaths = [
    path.resolve(__dirname, "../..", "data_power_consumption_sceaux.txt"),
    path.resolve(
      __dirname,
      "../../prevision-python",
      "data_power_consumption_sceaux.txt"
    ),
  ];
  const found = possiblePaths.find(fileExists);
  if (!found) {
    console.warn(
      "[seed] Arquivo data_power_consumption_sceaux.txt não encontrado. Pulando consumo."
    );
    return;
  }
  console.log(`[seed] Importando consumo de: ${found}`);
  const raw = await fs.promises.readFile(found, "utf-8");
  const lines = raw.split(/\r?\n/);
  const header = lines.shift() ?? "";
  const cols = header.split(";").map((c) => c.trim());
  // Expectadas: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3

  const limit = Number(process.env.SEED_POWER_LIMIT ?? 10000);
  const rows: {
    timestamp: Date;
    globalActivePower: number;
    globalReactivePower?: number | null;
    voltage?: number | null;
    globalIntensity?: number | null;
    subMetering1?: number | null;
    subMetering2?: number | null;
    subMetering3?: number | null;
  }[] = [];

  for (let i = 0; i < lines.length && rows.length < limit; i++) {
    const line = lines[i];
    if (!line) continue;
    const parts = line.split(";");
    if (parts.length < 9) continue;

    const dateStr = parts[0]?.trim();
    const timeStr = parts[1]?.trim();
    if (!dateStr || !timeStr) continue;
    // Format: dd/mm/yyyy HH:MM:SS
    const [d, m, y] = dateStr.split("/");
    const timestamp = new Date(`${y}-${m}-${d}T${timeStr}`);

    const globalActivePower = parseNumber(parts[2]);
    if (globalActivePower == null) continue; // skip invalid

    rows.push({
      timestamp,
      globalActivePower,
      globalReactivePower: parseNumber(parts[3]),
      voltage: parseNumber(parts[4]),
      globalIntensity: parseNumber(parts[5]),
      subMetering1: parseNumber(parts[6]),
      subMetering2: parseNumber(parts[7]),
      subMetering3: parseNumber(parts[8]),
    });
  }

  console.log(`[seed] Linhas de consumo preparadas: ${rows.length}`);

  const batchSize = 1000;
  for (let i = 0; i < rows.length; i += batchSize) {
    const batch = rows.slice(i, i + batchSize);
    await prisma.powerReading.createMany({ data: batch, skipDuplicates: true });
    console.log(`[seed] Consumo inserido: ${i + batch.length}/${rows.length}`);
  }
}

async function main() {
  try {
    await seedTemperature();
    await seedPower();
  } finally {
    await prisma.$disconnect();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
