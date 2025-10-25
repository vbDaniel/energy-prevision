import React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { StyledProgressRoot, StyledProgressIndicator } from "./styles";

export interface ProgressProps
  extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  value?: number;
}

const Progress = React.forwardRef<
  React.ComponentRef<typeof ProgressPrimitive.Root>,
  ProgressProps
>(({ value = 0, ...props }, ref) => (
  <StyledProgressRoot ref={ref} {...props}>
    <StyledProgressIndicator
      style={{ transform: `translateX(-${100 - value}%)` }}
    />
  </StyledProgressRoot>
));

Progress.displayName = "Progress";

export default Progress;
