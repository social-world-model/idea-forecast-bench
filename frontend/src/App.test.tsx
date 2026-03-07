import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Routes: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Route: ({ element }: { element: React.ReactElement }) => element,
  useNavigate: () => () => undefined,
  useLocation: () => ({ pathname: '/' }),
}), { virtual: true });

test('renders navigation brand', () => {
  render(<App />);
  const brandElement = screen.getByText(/Idea Benchmark/i);
  expect(brandElement).toBeInTheDocument();
});
