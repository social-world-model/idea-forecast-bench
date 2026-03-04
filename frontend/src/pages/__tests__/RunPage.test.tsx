import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import RunPage from '../RunPage';

describe('RunPage', () => {
  test('renders latest run in read-only mode', async () => {
    const fetchMock = jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        runs: [
          {
            run_id: 'abc12345-1234-5678-9012-abcdefabcdef',
            status: 'success',
            keywords: ['a'],
            n: 1,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            ideas_count: 1,
          },
        ],
      }),
    } as Response);

    render(
      <MemoryRouter>
        <RunPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Latest Run/i)).toBeInTheDocument();
      expect(screen.getByText(/success/i)).toBeInTheDocument();
      expect(screen.getByText(/read-only run monitor/i)).toBeInTheDocument();
    });

    fetchMock.mockRestore();
  });
});
