import React from "react";

/** A render error must never leave the user with a blank page. */
export default class ErrorBoundary extends React.Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error("UI error:", error, info?.componentStack); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="app">
        <div className="banner bad">
          Something went wrong displaying this screen: {String(this.state.error.message || this.state.error)}
        </div>
        <button className="primary" onClick={() => this.setState({ error: null })}>Try again</button>{" "}
        <button className="primary" onClick={() => location.reload()}>Reload</button>
      </div>
    );
  }
}
