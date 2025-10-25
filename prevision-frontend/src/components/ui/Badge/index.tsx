import * as React from "react";
import { StyledBadge, type BadgeVariant } from "./styles";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: BadgeVariant;
}

function Badge({ variant = "default", children, ...props }: BadgeProps) {
  return (
    <StyledBadge $variant={variant} {...props}>
      {children}
    </StyledBadge>
  );
}

export const badgeVariants = {
  default: "default",
  secondary: "secondary",
  destructive: "destructive",
  outline: "outline",
} as const;

export default { Badge, badgeVariants };
