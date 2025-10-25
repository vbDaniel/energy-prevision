import styled from "styled-components";
import * as ProgressPrimitive from "@radix-ui/react-progress";

/* Raiz (fundo da barra) */
export const StyledProgressRoot = styled(ProgressPrimitive.Root)`
  position: relative;
  height: 16px;
  width: 100%;
  overflow: hidden;
  border-radius: 30px;
  background-color: hsl(var(--muted)); /* equivalente a bg-secondary */
`;

/* Indicador (parte preenchida) */
export const StyledProgressIndicator = styled(ProgressPrimitive.Indicator)`
  height: 100%;
  width: 100%;
  flex: 1;
  background-color: hsl(var(--primary)); /* equivalente a bg-primary */
  transition: transform 0.3s ease;
`;
