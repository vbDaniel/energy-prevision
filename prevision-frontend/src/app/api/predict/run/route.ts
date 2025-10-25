import { NextRequest, NextResponse } from "next/server";
import { prisma } from "src/lib/prisma";

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8000";

type RunBody = {
  lookback?: number; // default 168 (7 days hourly)
  steps?: number; // how many predictions to produce (records - lookback + 1)
};

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json().catch(() => ({}))) as RunBody;
    const lookback =
      body.lookback && body.lookback > 0 ? body.lookback : 24 * 7;
    const steps = body.steps && body.steps > 0 ? body.steps : 24; // default 24 predictions (1 day)

    // Fetch latest power readings
    const totalNeeded = lookback + steps - 1;
    const power = await prisma.powerReading.findMany({
      orderBy: { timestamp: "desc" },
      take: totalNeeded,
    });
    if (power.length < lookback) {
      return NextResponse.json(
        {
          error: `Registros de consumo insuficientes: necessários ${lookback}, disponíveis ${power.length}`,
        },
        { status: 400 }
      );
    }
    // Reverse to chronological
    const powerChrono = power.reverse();

    // Fetch temperatures for the timeframe
    const startTs = powerChrono[0].timestamp;
    const endTs = powerChrono[powerChrono.length - 1].timestamp;
    const temps = await prisma.temperatureReading.findMany({
      where: { timestamp: { gte: startTs, lte: endTs } },
      orderBy: { timestamp: "asc" },
    });

    // Map temperatures by timestamp for quick nearest lookup
    const tempArr = temps.map((t: { timestamp: Date; temp: number }) => ({
      ts: t.timestamp.getTime(),
      v: t.temp,
    }));

    function nearestTemp(ts: Date): number | null {
      if (tempArr.length === 0) return null;
      const target = ts.getTime();
      // Binary search
      let lo = 0,
        hi = tempArr.length - 1;
      while (lo <= hi) {
        const mid = Math.floor((lo + hi) / 2);
        const midTs = tempArr[mid].ts;
        if (midTs === target) {
          return tempArr[mid].v;
        } else if (midTs < target) {
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      const cand1 = tempArr[Math.max(0, hi)];
      const cand2 = tempArr[Math.min(tempArr.length - 1, lo)];
      const dist1 = Math.abs(cand1.ts - target);
      const dist2 = Math.abs(cand2.ts - target);
      return dist1 <= dist2 ? cand1.v : cand2.v;
    }

    // Build records payload for Python
    const records = powerChrono.map(
      (p: { timestamp: Date; globalActivePower: number }) => ({
        datetime: p.timestamp.toISOString(),
        Global_active_power: p.globalActivePower,
        temp: nearestTemp(p.timestamp),
        // hour/dayofweek/month/holiday serão gerados no Python se ausentes
      })
    );

    const res = await fetch(`${PYTHON_API_URL}/predict/energy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records, lookback }),
    });

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: "Falha ao chamar API Python", detail: errText },
        { status: 502 }
      );
    }

    const data = (await res.json()) as {
      lookback: number;
      count: number;
      predictions: {
        datetime: string | null;
        real_kW: number;
        previsto_kW: number;
        residuo_kW: number;
      }[];
    };

    // Persist predictions to DB
    const inputStart = startTs;
    const inputEnd = endTs;
    const prediction = await prisma.prediction.create({
      data: {
        lookback,
        inputStart,
        inputEnd,
        count: data.count,
        status: "completed",
        modelName: "gru_energy_model_best",
        points: {
          createMany: {
            data: data.predictions.map((p) => ({
              timestamp: p.datetime ? new Date(p.datetime) : new Date(),
              realKw: isFinite(p.real_kW) ? p.real_kW : null,
              previstoKw: p.previsto_kW,
              residuoKw: isFinite(p.residuo_kW) ? p.residuo_kW : null,
            })),
            skipDuplicates: true,
          },
        },
      },
    });

    return NextResponse.json({
      predictionId: prediction.id,
      count: prediction.count,
    });
  } catch (e: any) {
    console.error(e);
    return NextResponse.json(
      { error: "Erro interno", detail: String(e?.message || e) },
      { status: 500 }
    );
  }
}
