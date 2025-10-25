import * as React from "react";
import {
  StyledAlert,
  StyledAlertTitle,
  StyledAlertDescription,
  AlertVariant,
} from "./styles";

export interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ variant = "default", ...props }, ref) => (
    <StyledAlert ref={ref} role="alert" variant={variant} {...props} />
  )
);
Alert.displayName = "Alert";

export interface AlertTitleProps
  extends React.HTMLAttributes<HTMLHeadingElement> {}

const AlertTitle = React.forwardRef<HTMLHeadingElement, AlertTitleProps>(
  ({ ...props }, ref) => <StyledAlertTitle ref={ref} {...props} />
);
AlertTitle.displayName = "AlertTitle";

export interface AlertDescriptionProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const AlertDescription = React.forwardRef<
  HTMLDivElement,
  AlertDescriptionProps
>(({ ...props }, ref) => <StyledAlertDescription ref={ref} {...props} />);
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
