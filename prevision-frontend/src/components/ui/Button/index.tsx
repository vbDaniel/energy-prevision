import React from "react";
import { Slot } from "@radix-ui/react-slot";
import { StyledButton } from "./styles";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?:
    | "default"
    | "destructive"
    | "outline"
    | "secondary"
    | "ghost"
    | "link";
  size?: "default" | "sm" | "lg" | "icon";
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { asChild = false, variant = "default", size = "default", ...props },
    ref
  ) => {
    const Comp = asChild ? Slot : "button";
    return (
      <StyledButton
        as={Comp}
        ref={ref}
        $variant={variant}
        $size={size}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";

export default Button;
