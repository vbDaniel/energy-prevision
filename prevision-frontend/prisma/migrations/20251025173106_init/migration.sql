-- CreateTable
CREATE TABLE "TemperatureReading" (
    "id" SERIAL NOT NULL,
    "timestamp" TIMESTAMP(6) NOT NULL,
    "temp" DOUBLE PRECISION NOT NULL,
    "location" TEXT DEFAULT 'Sceaux',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "TemperatureReading_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PowerReading" (
    "id" SERIAL NOT NULL,
    "timestamp" TIMESTAMP(6) NOT NULL,
    "globalActivePower" DOUBLE PRECISION NOT NULL,
    "globalReactivePower" DOUBLE PRECISION,
    "voltage" DOUBLE PRECISION,
    "globalIntensity" DOUBLE PRECISION,
    "subMetering1" DOUBLE PRECISION,
    "subMetering2" DOUBLE PRECISION,
    "subMetering3" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PowerReading_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Prediction" (
    "id" SERIAL NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lookback" INTEGER NOT NULL,
    "inputStart" TIMESTAMP(3) NOT NULL,
    "inputEnd" TIMESTAMP(3) NOT NULL,
    "modelName" TEXT DEFAULT 'gru_energy_model_best',
    "modelPath" TEXT,
    "status" TEXT NOT NULL DEFAULT 'completed',
    "count" INTEGER NOT NULL,

    CONSTRAINT "Prediction_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PredictionPoint" (
    "id" SERIAL NOT NULL,
    "predictionId" INTEGER NOT NULL,
    "timestamp" TIMESTAMP(6) NOT NULL,
    "realKw" DOUBLE PRECISION,
    "previstoKw" DOUBLE PRECISION NOT NULL,
    "residuoKw" DOUBLE PRECISION,

    CONSTRAINT "PredictionPoint_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "TemperatureReading_timestamp_idx" ON "TemperatureReading"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "TemperatureReading_timestamp_location_key" ON "TemperatureReading"("timestamp", "location");

-- CreateIndex
CREATE INDEX "PowerReading_timestamp_idx" ON "PowerReading"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "PowerReading_timestamp_key" ON "PowerReading"("timestamp");

-- CreateIndex
CREATE INDEX "PredictionPoint_predictionId_idx" ON "PredictionPoint"("predictionId");

-- CreateIndex
CREATE INDEX "PredictionPoint_timestamp_idx" ON "PredictionPoint"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "PredictionPoint_predictionId_timestamp_key" ON "PredictionPoint"("predictionId", "timestamp");

-- AddForeignKey
ALTER TABLE "PredictionPoint" ADD CONSTRAINT "PredictionPoint_predictionId_fkey" FOREIGN KEY ("predictionId") REFERENCES "Prediction"("id") ON DELETE CASCADE ON UPDATE CASCADE;
