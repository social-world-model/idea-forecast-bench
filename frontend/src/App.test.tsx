import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders navigation brand', () => {
  render(<App />);
  const brandElement = screen.getByText(/Idea Benchmark/i);
  expect(brandElement).toBeInTheDocument();
});
